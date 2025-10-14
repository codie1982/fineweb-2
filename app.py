# -*- coding: utf-8 -*-
"""
Multiprocessing ingest + resume (processed_files) + safe shutdown + atomic checkpoint + dup guard + heartbeat.
"""

import os, glob, queue, time, datetime, sys, signal, atexit, json, hashlib, gc
from typing import List, Optional, Set
from multiprocessing import Process, Queue, Event, set_start_method

import numpy as np
import pyarrow.dataset as ds
import faiss
import torch
from sentence_transformers import SentenceTransformer

# --- colorama (opsiyonel) ---
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except Exception:
    COLOR = False
    class _Dummy:
        def __getattr__(self, k): return ""
    Fore = Style = _Dummy()

# --- senin FAISS sarmalayıcın ---
from embedding_index import EmbeddingIndex

# ============== AYARLAR ==============
ROOT_DIR         = "./fineweb-2/data/tur_Latn/train"
INDEX_PATH       = "faiss.index"
META_PATH        = "meta.json"
MODEL_NAME       = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

N_WORKERS        = 12
LIMIT_TOTAL      = 100
BATCH_READ       = 4096
BATCH_ENCODE     = 24
MIN_CHARS        = 80
MAX_CHARS        = 8000
DOC_TYPE         = "fineweb2"
MAX_TEXT_CHARS   = 1500
MAX_SEG_LENGTH   = 128
TEXT_CANDS       = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS        = ["url","source_url","link","origin","canonical_url"]

Q_R2W_SIZE       = 2000
Q_W2WR_SIZE      = 2000

FLUSH_EVERY      = 2000
FLUSH_INTERVAL_S = 30

PROCESSED_FILE   = "processed_files.txt"
LOG_FILE         = None
WRITER_POISON    = "__WRITER_POISON__"   # 🔥 tek seferlik writer kapatma sinyali
# =====================================


def processed_file_path() -> str:
    return os.path.abspath(PROCESSED_FILE)

def ensure_processed_file_exists():
    path = processed_file_path()
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
    return path

def load_processed() -> Set[str]:
    path = ensure_processed_file_exists()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(l.strip() for l in f if l.strip())
    except Exception as e:
        log(f"⚠️ processed_files okunamadı: {e}", "SYS", Fore.YELLOW)
        return set()

def mark_processed(path: str):
    pf = processed_file_path()
    try:
        with open(pf, "a", encoding="utf-8") as f:
            f.write(path + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        log(f"⚠️ processed_files yazılamadı: {e}", "READER", Fore.YELLOW)


# -------------- LOG --------------
def _stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg: str, tag="SYS", color=Fore.CYAN):
    line = f"[{_stamp()}] [{tag}] {msg}"
    if COLOR: print(f"{color}{line}{Style.RESET_ALL}")
    else: print(line)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# -------------- UTIL --------------
def list_parquets(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))

def pick_col(schema_names: List[str], cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in schema_names:
            return c
    return None

def text_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def flush_atomic(store: EmbeddingIndex):
    meta_tmp = store.meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in store.meta.items()}, f, ensure_ascii=False)
    os.replace(meta_tmp, store.meta_path)

    idx_tmp = store.index_path + ".tmp"
    faiss.write_index(store.index, idx_tmp)
    os.replace(idx_tmp, store.index_path)


# ----- Global model -----
MODEL: Optional[SentenceTransformer] = None
def preload_model():
    global MODEL
    if MODEL is None:
        log("Model yükleniyor…", "SYS", Fore.CYAN)
        MODEL = SentenceTransformer(MODEL_NAME)
        MODEL.max_seq_length = MAX_SEG_LENGTH
        MODEL.eval()
        log("Model hazır.", "SYS", Fore.CYAN)


# -------------- READER --------------
def reader(files: List[str], out_q: Queue, stop_event: Event):
    sent = 0
    total_limit = LIMIT_TOTAL if isinstance(LIMIT_TOTAL, int) else float("inf")
    seen_files = load_processed()
    todo = [p for p in files if p not in seen_files]
    log(f"Reader başladı: {len(files)} dosya (atlanan={len(seen_files)}, işlenecek={len(todo)}).", "READER", Fore.BLUE)

    for path in todo:
        if stop_event.is_set() or sent >= total_limit:
            break
        try:
            ds_ = ds.dataset(path, format="parquet")
        except Exception as e:
            log(f"❌ okunamadı: {path} -> {e}", "READER", Fore.RED)
            mark_processed(path)
            continue

        names = ds_.schema.names
        tcol  = pick_col(names, TEXT_CANDS)
        ucol  = pick_col(names, URL_CANDS)
        if not tcol:
            log(f"⚠️ metin kolonu yok, atla: {path}", "READER", Fore.YELLOW)
            mark_processed(path)
            continue

        scanner = ds_.scanner(columns=[tcol] + ([ucol] if ucol else []), batch_size=BATCH_READ, use_threads=True)
        for b in scanner.to_batches():
            if stop_event.is_set(): break
            d = b.to_pydict()
            texts = d.get(tcol, [])
            urls  = d.get(ucol, []) if ucol else [None]*len(texts)
            for i, t in enumerate(texts):
                if stop_event.is_set() or sent >= total_limit: break
                if not t: continue
                s = str(t).strip()
                if MAX_TEXT_CHARS:
                    s = s[:MAX_TEXT_CHARS]
                if not (MIN_CHARS <= len(s) <= MAX_CHARS): continue
                u = urls[i] if urls and urls[i] else f"http://fw2.local/{os.path.basename(path)}-{i}"
                out_q.put((s, u))
                sent += 1
        mark_processed(path)

    for _ in range(N_WORKERS):
        out_q.put(None)
    log(f"✅ Reader bitti. Kuyruğa {sent} örnek gönderildi.", "READER", Fore.BLUE)


# -------------- WORKER --------------
def worker(in_q, out_q, stop_event, wid: int):
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME)
        MODEL.eval()

    log(f"Worker-{wid} başladı.", f"W{wid}", Fore.MAGENTA)
    processed = 0
    batch_texts, batch_urls = [], []
    last_push_t = time.time()
    PARTIAL_FLUSH_S = 10

    while True:
        if stop_event.is_set():
            if batch_texts:
                vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE, show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
                out_q.put((vecs, batch_texts, batch_urls))
            out_q.put(None)
            break

        try:
            item = in_q.get(timeout=2)
        except queue.Empty:
            if batch_texts and (time.time() - last_push_t) >= PARTIAL_FLUSH_S:
                vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE, show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
                out_q.put((vecs, batch_texts, batch_urls))
                batch_texts, batch_urls = [], []
            continue

        if item is None:
            if batch_texts:
                vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE, show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
                out_q.put((vecs, batch_texts, batch_urls))
            out_q.put(None)
            break

        text, url = item
        batch_texts.append(text)
        batch_urls.append(url)
        if len(batch_texts) >= BATCH_ENCODE:
            vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE, show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
            out_q.put((vecs, batch_texts, batch_urls))
            batch_texts, batch_urls = [], []
            last_push_t = time.time()

    log(f"Worker-{wid} tamamlandı. İşlenen: {processed}", f"W{wid}", Fore.MAGENTA)


# -------------- WRITER --------------
def writer(in_q: Queue, stop_event: Event):
    try:
        store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)
        seen_sha1 = { (v.get("metadata") or {}).get("sha1") for v in store.meta.values() if (v.get("metadata") or {}).get("sha1") }
        finished = 0
        total_added = 0
        t0 = time.time()

        log("Writer başladı.", "WRITER", Fore.GREEN)

        while True:
            if finished >= N_WORKERS:
                break

            try:
                item = in_q.get(timeout=5)
            except queue.Empty:
                continue

            # 🔥 Parent’tan gelen kapatma sinyali
            if item == WRITER_POISON:
                break

            if item is None:
                finished += 1
                continue

            vecs, texts, urls = item
            vecs = np.asarray(vecs, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)

            keep_idx = []
            sha1_list = []
            for i, txt in enumerate(texts):
                h = text_sha1(txt)
                if h in seen_sha1:
                    continue
                keep_idx.append(i)
                sha1_list.append(h)

            if not keep_idx:
                continue

            vecs = vecs[keep_idx, :]
            texts = [texts[i] for i in keep_idx]
            urls  = [urls[i] for i in keep_idx]

            dim = vecs.shape[1]
            with store._lock:
                store._ensure_index(dim)
                start = store._next_int_id
                ids = np.arange(start, start + len(texts), dtype=np.int64)
                store._next_int_id += len(texts)
                store.index.add_with_ids(vecs, ids)
                for j, fid in enumerate(ids):
                    store.meta[int(fid)] = {
                        "external_id": os.urandom(8).hex(),
                        "text": texts[j],
                        "metadata": {"doc_type": DOC_TYPE, "url": urls[j], "sha1": sha1_list[j]},
                    }
                    seen_sha1.add(sha1_list[j])

            total_added += len(texts)

        flush_atomic(store)
        log(f"✅ Writer tamamlandı. Toplam {total_added} vektör | {time.time()-t0:.1f}s", "WRITER", Fore.GREEN)

    except Exception as e:
        log(f"⚠️ Writer hatası: {e}", "WRITER", Fore.YELLOW)


# -------------- MAIN --------------
def main():
    try:
        set_start_method("fork", force=True)
    except RuntimeError:
        pass

    pf = ensure_processed_file_exists()
    log(f"Resume dosyası: {pf}", "SYS", Fore.CYAN)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    preload_model()

    files = list_parquets(ROOT_DIR)
    if not files:
        log(f"❌ Parquet bulunamadı: {ROOT_DIR}", "SYS", Fore.RED)
        sys.exit(1)

    stop_event = Event()
    q_r2w  = Queue(maxsize=Q_R2W_SIZE)
    q_w2wr = Queue(maxsize=Q_W2WR_SIZE)

    p_reader = Process(target=reader, args=(files, q_r2w, stop_event), daemon=True)
    p_writer = Process(target=writer, args=(q_w2wr, stop_event), daemon=False)
    workers  = [Process(target=worker, args=(q_r2w, q_w2wr, stop_event, i+1), daemon=True) for i in range(N_WORKERS)]

    def graceful_shutdown(signum=None, frame=None):
        log(f"⚠️ Sinyal alındı ({signum}). Kapanış başlatılıyor…", "SYS", Fore.YELLOW)
        stop_event.set()
        try:
            for _ in range(N_WORKERS):
                q_r2w.put_nowait(None)
        except Exception:
            pass
        try:
            q_w2wr.put_nowait(WRITER_POISON)
        except Exception:
            pass

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(lambda: stop_event.set())

    t0 = time.time()
    p_writer.start()
    for p in workers: p.start()
    p_reader.start()

    try:
        p_reader.join()
        for p in workers: p.join()

        # ✅ writer’a tek seferlik poison pill gönder
        try:
            q_w2wr.put_nowait(WRITER_POISON)
        except Exception:
            pass

        p_writer.join()

    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        try:
            q_w2wr.put_nowait(WRITER_POISON)
        except Exception:
            pass
        p_writer.join(timeout=10)
        if p_writer.is_alive():
            p_writer.terminate()

    finally:
        log(f"🏁 Toplam süre: {time.time()-t0:.1f}s", "SYS", Fore.CYAN)
        log(f"FAISS: {INDEX_PATH} | META: {META_PATH}", "SYS", Fore.CYAN)


if __name__ == "__main__":
    main()
