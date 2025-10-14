# -*- coding: utf-8 -*-
"""
Multiprocessing ingest + resume (processed_files) + safe shutdown + atomic checkpoint + dup guard + heartbeat.
"""

import os, glob, queue, time, datetime, sys, signal, atexit, json, hashlib
from typing import List, Optional, Set
from multiprocessing import Process, Queue, Event, set_start_method

import numpy as np
import pyarrow.dataset as ds
import faiss
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
LIMIT_TOTAL      = 100
BATCH_READ       = 4096
MIN_CHARS        = 80
MAX_CHARS        = 8000
DOC_TYPE         = "fineweb2"
MAX_TEXT_CHARS   = None   # <-- chunking zaten yapılacak, burada kırpmaya gerek yok
TEXT_CANDS       = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS        = ["url","source_url","link","origin","canonical_url"]

Q_R2W_SIZE       = 2000
Q_W2WR_SIZE      = 2000

PROCESSED_FILE   = "processed_files.txt"
LOG_FILE         = None
WRITER_POISON    = "__WRITER_POISON__"
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
                if not (MIN_CHARS <= len(s) <= MAX_CHARS): continue
                u = urls[i] if urls and urls[i] else f"http://fw2.local/{os.path.basename(path)}-{i}"
                out_q.put((s, u))
                sent += 1
        mark_processed(path)

    for _ in range(N_WORKERS):
        out_q.put(None)
    log(f"✅ Reader bitti. Kuyruğa {sent} örnek gönderildi.", "READER", Fore.BLUE)


# -------------- WORKER (sadece iletici) --------------
def worker(in_q, out_q, stop_event, wid: int):
    log(f"Worker-{wid} başladı.", f"W{wid}", Fore.MAGENTA)
    while True:
        if stop_event.is_set():
            out_q.put(None)
            break

        try:
            item = in_q.get(timeout=2)
        except queue.Empty:
            continue

        if item is None:
            out_q.put(None)
            break

        text, url = item
        if text.strip():
            out_q.put((None, [text], [url]))

    log(f"Worker-{wid} tamamlandı.", f"W{wid}", Fore.MAGENTA)


# -------------- WRITER --------------
def writer(in_q: Queue, stop_event: Event):
    try:
        store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)
        seen_sha1 = { (v.get("metadata") or {}).get("full_text_sha1") for v in store.meta.values() if (v.get("metadata") or {}).get("full_text_sha1") }
        finished = 0
        total_added = 0
        t0 = time.time()

        log("Writer başladı (chunk destekli).", "WRITER", Fore.GREEN)

        while True:
            if finished >= N_WORKERS:
                break

            try:
                item = in_q.get(timeout=5)
            except queue.Empty:
                continue

            if item == WRITER_POISON:
                break

            if item is None:
                finished += 1
                continue

            _, texts, urls = item

            for i, txt in enumerate(texts):
                if not txt.strip():
                    continue

                full_hash = text_sha1(txt)
                if full_hash in seen_sha1:
                    continue

                metadata = {
                    "doc_type": DOC_TYPE,
                    "url": urls[i],
                    "full_text_sha1": full_hash,
                    "full_text": txt
                }

                results = store.upsert_vector(
                    text=txt,
                    metadata=metadata,
                    chunk_size=1500,
                    overlap=200
                )

                seen_sha1.add(full_hash)
                total_added += len(results)

        flush_atomic(store)
        log(f"✅ Writer tamamlandı. Toplam {total_added} chunk eklendi | Süre: {time.time()-t0:.1f}s", "WRITER", Fore.GREEN)

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
            q_w2wr.put_nowait(WRITER_POISON)
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
