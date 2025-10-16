# -*- coding: utf-8 -*-
"""
Multiprocessing ingest + resume (processed_files) + safe shutdown + atomic checkpoint + dup guard + heartbeat.
Paralel encode (worker), toplu FAISS add (writer), ayrıntılı log.
"""

import os, glob, queue, time, datetime, sys, signal, atexit, json, hashlib
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

# --- FAISS sarmalayıcı ---
from embedding_index import EmbeddingIndex

# ============== AYARLAR ==============
ROOT_DIR         = "./fineweb-2/data/tur_Latn/train"
INDEX_PATH       = "faiss.index"
META_PATH        = "meta.json"
MODEL_NAME       = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

N_WORKERS        = 12
LIMIT_TOTAL      = 100              # Tüm veri için None ya da çok büyük sayı ver
BATCH_READ       = 4096
MIN_CHARS        = 80
MAX_CHARS        = 8000
DOC_TYPE         = "fineweb2"
MAX_TEXT_CHARS   = None             # kırpma yok (chunk/encode worker’da)
TEXT_CANDS       = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS        = ["url","source_url","link","origin","canonical_url"]

Q_R2W_SIZE       = 2000
Q_W2WR_SIZE      = 2000

LOG_FILE         = None             # örn: "logs/ingest.log"
WRITER_POISON    = "__WRITER_POISON__"
# --- throughput ayarları ---
READ_DISPATCH    = 4096   # reader -> worker tek mesajda kaç satır gönderecek
CHUNK_SIZE       = 1500   # chunk uzunluğu (karakter)
CHUNK_OVERLAP    = 200    # chunk overlap (karakter)
ENC_BATCH        = 512    # encode batch size (GPU: 256–1024, CPU: 64–256)

# =====================================

# --------- Global Model (fork öncesi preload için) ---------
MODEL = None

def _stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg: str, tag="SYS", color=Fore.CYAN):
    line = f"[{_stamp()}] [{tag}] {msg}"
    if COLOR: print(f"{color}{line}{Style.RESET_ALL}", flush=True)
    else: print(line, flush=True)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def _fmt_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB","PB","EB"]:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}EB"

def processed_file_path() -> str:
    return os.path.abspath("processed_files.txt")

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
    """meta.json ve faiss.index'i temp'e yazıp atomik replace."""
    try:
        meta_tmp = store.meta_path + ".tmp"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in store.meta.items()}, f, ensure_ascii=False)
        os.replace(meta_tmp, store.meta_path)

        idx_tmp = store.index_path + ".tmp"
        faiss.write_index(store.index, idx_tmp)
        os.replace(idx_tmp, store.index_path)
    except Exception as e:
        log(f"⚠️ flush_atomic hata: {e}", "WRITER", Fore.YELLOW)

def preload_model():
    """Worker’ları fork etmeden ÖNCE modeli yükle (copy-on-write ile paylaşım)."""
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = SentenceTransformer(MODEL_NAME, device=device)
    MODEL.eval()
    log(f"Encoder hazır: {MODEL_NAME} | device={device}", "SYS", Fore.CYAN)
    return MODEL

# -------------- READER --------------
def reader(files: List[str], out_q: Queue, stop_event: Event):
    sent = 0
    total_limit = LIMIT_TOTAL if isinstance(LIMIT_TOTAL, int) else float("inf")
    seen_files = load_processed()
    todo = [p for p in files if p not in seen_files]
    log(f"Reader başladı: {len(files)} dosya (atlanan={len(seen_files)}, işlenecek={len(todo)}).", "READER", Fore.BLUE)

    batch_T, batch_U = [], []
    def flush_reader_batch():
        nonlocal batch_T, batch_U, sent
        if not batch_T:
            return
        out_q.put((batch_T, batch_U))      # 🎯 tek mesaj = binlerce satır
        sent += len(batch_T)
        batch_T, batch_U = [], []

    for path in todo:
        log(f"[FILE] reading: {path}", "READER", Fore.BLUE)
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

        scanner = ds_.scanner(columns=[tcol] + ([ucol] if ucol else []),
                              batch_size=BATCH_READ, use_threads=True)
        for b in scanner.to_batches():
            if stop_event.is_set(): break
            d = b.to_pydict()
            texts = d.get(tcol, [])
            urls  = d.get(ucol, []) if ucol else [None]*len(texts)
            for i, t in enumerate(texts):
                if stop_event.is_set() or sent >= total_limit: break
                if not t: continue
                s = str(t).strip()
                if not (MIN_CHARS <= len(s) <= MAX_CHARS): continue
                u = urls[i] if urls and urls[i] else f"http://fw2.local/{os.path.basename(path)}-{i}"

                # 🔹 tek tek put etmek yerine paketle
                batch_T.append(s)
                batch_U.append(u)

                if len(batch_T) >= READ_DISPATCH:
                    flush_reader_batch()

        flush_reader_batch()     # dosya bitişinde elde kalanları gönder
        mark_processed(path)

    # worker'lara bitiş sinyali
    for _ in range(N_WORKERS):
        out_q.put(None)
    log(f"✅ Reader bitti. Kuyruğa {sent} satır gönderildi (paketli).", "READER", Fore.BLUE)
# -------------- WORKER: encode + ilet --------------
def _chunk_simple(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    n = len(s)
    step = max(1, size - overlap)
    for start in range(0, n, step):
        end = min(n, start + size)
        yield s[start:end]

def worker(in_q, out_q, stop_event, wid: int):
    import gc, time
    try: torch.set_num_threads(1)
    except Exception: pass
    try: faiss.omp_set_num_threads(1)
    except Exception: pass

    device = "cuda" if torch.cuda.is_available() else "cpu"

    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME, device=device)
        MODEL.eval()
        if device == "cuda":
            try: MODEL = MODEL.half()
            except Exception: pass
    model = MODEL  # paylaşılan nesneyi kullan

    # GPU'da mümkünse fp16 (VRAM/hız kazancı)
    if device == "cuda":
        try: model = model.half()
        except Exception: pass

    log(f"Worker-{wid} başladı. device={device}", f"W{wid}", Fore.MAGENTA)

    while True:
        if stop_event.is_set():
            out_q.put(None)
            break

        try:
            item = in_q.get(timeout=1.5)
        except queue.Empty:
            continue

        if item is None:
            out_q.put(None)
            break

        # 🎯 reader'dan paket gelir: (list_of_texts, list_of_urls)
        texts, urls = item

        # 1) paket içindeki tüm metinleri chunk'la → tek dev batch
        batch_texts, batch_urls = [], []
        for s, u in zip(texts, urls):
            if not s: continue
            s = s.strip()
            if not s: continue
            for ch in _chunk_simple(s):
                if len(ch) < MIN_CHARS:  # çok kısa chunk'ı at
                    continue
                batch_texts.append(ch)
                batch_urls.append(u)

        if not batch_texts:
            continue

        # 2) büyük batch encode (parça parça)
        vecs_out = []
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(batch_texts), ENC_BATCH):
                sub = batch_texts[i:i+ENC_BATCH]
                v = model.encode(
                    sub,
                    batch_size=ENC_BATCH,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                vecs_out.append(np.asarray(v, dtype=np.float32))

        vecs = np.vstack(vecs_out) if len(vecs_out) > 1 else vecs_out[0]
        dt = time.time() - t0

        # 3) writer'a tek vuruşta gönder
        out_q.put((vecs, batch_texts, batch_urls))
        log(f"Worker-{wid}: sent {vecs.shape[0]} chunks in {dt*1000:.0f}ms "
            f"(~{vecs.shape[0]/max(dt,1e-6):.1f} vec/s)", f"W{wid}", Fore.MAGENTA)

        # bellek toparla
        del vecs_out; gc.collect()

    log(f"Worker-{wid} tamamlandı.", f"W{wid}", Fore.MAGENTA)

# -------------- WRITER: toplu FAISS add + atomik flush --------------
def writer(in_q: Queue, stop_event: Event):
    try:
        store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)

        finished = 0
        total_chunks_added = 0
        t0 = time.time()
        last_hb = t0
        last_rate_n = 0
        HB_EVERY_S = 30
        RATE_EVERY = 10000

        ACC_VEC, ACC_TXT, ACC_URL = [], [], []
        FLUSH_ADD_EVERY = 2048   # 2k vektörde bir FAISS.add

        log("Writer başladı (sadece add).", "WRITER", Fore.GREEN)

        def flush_add():
            nonlocal ACC_VEC, ACC_TXT, ACC_URL, total_chunks_added
            if not ACC_VEC:
                return
            vecs = np.vstack(ACC_VEC).astype(np.float32, copy=False)
            dim = vecs.shape[1]
            with store._lock:
                store._ensure_index(dim)
                store._normalize(vecs)
                start_id = store._next_int_id
                ids = np.arange(start_id, start_id + vecs.shape[0], dtype=np.int64)
                store._next_int_id += vecs.shape[0]
                store.index.add_with_ids(vecs, ids)
                for j, fid in enumerate(ids):
                    store.meta[int(fid)] = {
                        "external_id": os.urandom(8).hex(),
                        # metni yazmıyoruz → meta.json küçük kalır
                        "metadata": {
                            "doc_type": DOC_TYPE,
                            "url": ACC_URL[j],
                        },
                    }
            total_chunks_added += vecs.shape[0]
            ACC_VEC.clear(); ACC_TXT.clear(); ACC_URL.clear()

        while True:
            now = time.time()
            if now - last_hb >= HB_EVERY_S:
                ntotal = store.index.ntotal if store.index else 0
                meta_size = os.path.getsize(store.meta_path) if os.path.exists(store.meta_path) else 0
                idx_size  = os.path.getsize(store.index_path) if os.path.exists(store.index_path) else 0
                log(f"[HB] chunks={total_chunks_added} ntotal={ntotal} "
                    f"meta={_fmt_bytes(meta_size)} index={_fmt_bytes(idx_size)}",
                    "WRITER", Fore.CYAN)
                last_hb = now

            if total_chunks_added - last_rate_n >= RATE_EVERY:
                dt = max(now - t0, 1e-6)
                rps = total_chunks_added / dt
                log(f"[RATE] chunks={total_chunks_added} | {rps:.1f} vec/s", "WRITER", Fore.CYAN)
                last_rate_n = total_chunks_added

            if finished >= N_WORKERS:
                break

            try:
                item = in_q.get(timeout=5)
            except queue.Empty:
                if ACC_VEC:
                    flush_add()
                continue

            if item == WRITER_POISON:
                break
            if item is None:
                finished += 1
                continue

            vecs, texts, urls = item
            ACC_VEC.append(np.asarray(vecs, dtype=np.float32))
            ACC_TXT.extend(texts)
            ACC_URL.extend(urls)

            if sum(v.shape[0] for v in ACC_VEC) >= FLUSH_ADD_EVERY:
                flush_add()

        flush_add()
        flush_atomic(store)

        dt = time.time() - t0
        rps = total_chunks_added / max(dt, 1e-6)
        ntotal = store.index.ntotal if store.index else 0
        meta_size = os.path.getsize(store.meta_path) if os.path.exists(store.meta_path) else 0
        idx_size  = os.path.getsize(store.index_path) if os.path.exists(store.index_path) else 0
        log(f"✅ Writer tamamlandı. chunks={total_chunks_added} | {dt:.1f}s ~{rps:.1f} vec/s "
            f"| ntotal={ntotal} meta={_fmt_bytes(meta_size)} index={_fmt_bytes(idx_size)}",
            "WRITER", Fore.GREEN)

    except Exception as e:
        log(f"⚠️ Writer hatası: {e}", "WRITER", Fore.YELLOW)

# -------------- MAIN --------------
def main():
    try:
        set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # ÖNEMLİ: fork’tan önce modeli yükle (Linux/Unix’te bellek paylaşılır)
    preload_model()

    pf = ensure_processed_file_exists()
    log(f"Resume dosyası: {pf}", "SYS", Fore.CYAN)

    files = list_parquets(ROOT_DIR)
    if not files:
        log(f"❌ Parquet bulunamadı: {ROOT_DIR}", "SYS", Fore.RED)
        sys.exit(1)

    stop_event = Event()
    q_r2w  = Queue(maxsize=Q_R2W_SIZE)
    q_w2wr = Queue(maxsize=Q_W2WR_SIZE)

    p_reader = Process(target=reader, args=(files, q_r2w, stop_event), daemon=True)
    p_writer = Process(target=writer, args=(q_w2wr, stop_event), daemon=False)
    workers  = [Process(target=worker, args=(q_r2w, q_w2wr, stop_event, i+1), daemon=True)
                for i in range(N_WORKERS)]

    def graceful_shutdown(signum=None, frame=None):
        log(f"⚠️ Sinyal alındı ({signum}). Kapanış başlatılıyor…", "SYS", Fore.YELLOW)
        stop_event.set()
        try:
            for _ in range(N_WORKERS): q_r2w.put_nowait(None)
        except Exception: pass
        try:
            q_w2wr.put_nowait(WRITER_POISON)
        except Exception: pass

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

        try:
            q_w2wr.put_nowait(WRITER_POISON)  # tek writer olduğu için 1 poison yeter
        except Exception:
            pass

        p_writer.join()

    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)
        for p in workers:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
        try:
            q_w2wr.put_nowait(WRITER_POISON)
        except Exception: pass
        p_writer.join(timeout=10)
        if p_writer.is_alive(): p_writer.terminate()

    finally:
        log(f"🏁 Toplam süre: {time.time()-t0:.1f}s", "SYS", Fore.CYAN)
        log(f"FAISS: {INDEX_PATH} | META: {META_PATH}", "SYS", Fore.CYAN)

if __name__ == "__main__":
    main()
