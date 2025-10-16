# -*- coding: utf-8 -*-
"""
Simple & fast multiprocess ingest
- Reader: paketli üretim
- Worker: chunk + büyük batch encode (numpy float32)
- Writer: toplu FAISS add + periyodik heartbeat + final atomic flush
"""

import os, glob, queue, time, sys, signal, atexit, json, hashlib
from typing import List, Optional, Set
from multiprocessing import Process, Queue, Event, set_start_method, cpu_count

import numpy as np
import pyarrow.dataset as ds
import faiss
import torch
from sentence_transformers import SentenceTransformer

# ---- Mini color ----
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except Exception:
    COLOR = False
    class _D:  # fallback
        def __getattr__(self, k): return ""
    Fore = Style = _D()

# ---- Proje sınıfı ----
from embedding_index import EmbeddingIndex

# ================== AYARLAR ==================
ROOT_DIR   = "./fineweb-2/data/tur_Latn/train"
INDEX_PATH = "faiss.index"
META_PATH  = "meta.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CPUS      = cpu_count()
N_WORKERS = max(4, min(16, CPUS - 2))  # 2 çekirdek reader+writer’a

LIMIT_TOTAL = None          # tüm veri için None
BATCH_READ  = 8192
MIN_CHARS   = 80
MAX_CHARS   = 8000
DOC_TYPE    = "fineweb2"

TEXT_CANDS  = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS   = ["url","source_url","link","origin","canonical_url"]

Q_R2W_SIZE  = N_WORKERS * 400
Q_W2WR_SIZE = N_WORKERS * 200

READ_DISPATCH = 8192        # reader->worker paket büyüklüğü
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200
ENC_BATCH     = 256         # CPU: 128–512 deneyebilirsin

FLUSH_ADD_EVERY = 2048      # writer add eşiği
HEARTBEAT_EVERY = 30         # saniye

WRITER_POISON = "__WRITER_POISON__"
# ============================================

MODEL: Optional[SentenceTransformer] = None  # fork öncesi preload

# ---- yardımcılar ----
def log(msg, tag="SYS", color=Fore.CYAN):
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    if COLOR: print(f"{color}{line}{Style.RESET_ALL}", flush=True)
    else: print(line, flush=True)

def list_parquets(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))

def pick_col(schema_names: List[str], cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in schema_names:
            return c
    return None

def processed_file_path() -> str:
    return os.path.abspath("processed_files.txt")

def ensure_processed_file_exists():
    p = processed_file_path()
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8"): pass
    return p

def load_processed() -> Set[str]:
    ensure_processed_file_exists()
    with open(processed_file_path(), "r", encoding="utf-8") as f:
        return set(l.strip() for l in f if l.strip())

def mark_processed(path: str):
    with open(processed_file_path(), "a", encoding="utf-8") as f:
        f.write(path + "\n")

def fmt_bytes(n: int) -> str:
    s = ["B","KB","MB","GB","TB","PB","EB"]
    i = 0; x = float(n)
    while x >= 1024 and i < len(s)-1:
        x /= 1024; i += 1
    return f"{x:.1f}{s[i]}"

def flush_atomic(store: EmbeddingIndex):
    # meta
    mt = store.meta_path + ".tmp"
    with open(mt, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in store.meta.items()}, f, ensure_ascii=False)
    os.replace(mt, store.meta_path)
    # index
    it = store.index_path + ".tmp"
    faiss.write_index(store.index, it)
    os.replace(it, store.index_path)

def preload_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    torch.set_num_threads(1)  # worker başına 1 thread
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = SentenceTransformer(MODEL_NAME, device=device)
    MODEL.eval()
    log(f"Encoder hazır: {MODEL_NAME} | device={device} | workers={N_WORKERS}")
    return MODEL

# ---- READER ----
def reader(files: List[str], out_q: Queue, stop_event: Event):
    total = 0
    seen  = load_processed()
    todo  = [p for p in files if p not in seen]
    log(f"Reader: {len(files)} dosya (skip={len(seen)}, todo={len(todo)})", "READER", Fore.BLUE)

    bufT: List[str] = []
    bufU: List[Optional[str]] = []

    def flush(force=False):
        nonlocal bufT, bufU, total
        if not bufT: return
        if not force and len(bufT) < READ_DISPATCH: return
        out_q.put((bufT, bufU))      # tek put, çok satır
        total += len(bufT)
        bufT, bufU = [], []

    for pidx, path in enumerate(todo):
        if stop_event.is_set(): break
        log(f"[{pidx+1}/{len(todo)}] read: {os.path.basename(path)}", "READER", Fore.BLUE)

        try:
            ds_ = ds.dataset(path, format="parquet")
        except Exception as e:
            log(f"okunamadı: {path} -> {e}", "READER", Fore.RED); mark_processed(path); continue

        names = ds_.schema.names
        tcol  = pick_col(names, TEXT_CANDS); ucol = pick_col(names, URL_CANDS)
        if not tcol:
            log(f"metin kolonu yok, atla: {path}", "READER", Fore.YELLOW); mark_processed(path); continue

        scanner = ds_.scanner(columns=[tcol] + ([ucol] if ucol else []),
                              batch_size=BATCH_READ, use_threads=True)

        for b in scanner.to_batches():
            if stop_event.is_set(): break
            d = b.to_pydict()
            texts = d.get(tcol, [])
            urls  = d.get(ucol, []) if ucol else [None]*len(texts)
            for i, t in enumerate(texts):
                if stop_event.is_set(): break
                if LIMIT_TOTAL and total >= LIMIT_TOTAL: break
                if not t: continue
                s = str(t).strip()
                if not (MIN_CHARS <= len(s) <= MAX_CHARS): continue
                u = urls[i] if urls and urls[i] else f"http://fw2.local/{os.path.basename(path)}-{i}"
                bufT.append(s); bufU.append(u)
                if len(bufT) >= READ_DISPATCH:
                    flush(force=True)
        flush(force=True)
        mark_processed(path)

    flush(force=True)
    for _ in range(N_WORKERS): out_q.put(None)
    log(f"Reader bitti. queue'ya {total} satır gönderildi.", "READER", Fore.BLUE)

# ---- WORKER ----
def _chunk_simple(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    n = len(s); step = max(1, size - overlap)
    for st in range(0, n, step):
        ed = min(n, st + size)
        yield s[st:ed]

def worker(in_q: Queue, out_q: Queue, stop_event: Event, wid: int):
    torch.set_num_threads(1)
    try: faiss.omp_set_num_threads(1)
    except Exception: pass

    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
        MODEL.eval()

    log(f"Worker-{wid} başladı", f"W{wid}", Fore.MAGENTA)

    last_log = time.time()
    while not stop_event.is_set():
        try:
            item = in_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break

        texts, urls = item

        # paket içindeki metinleri chunk’la
        chunks, churn = [], []
        for s, u in zip(texts, urls):
            if not s: continue
            s = s.strip()
            if not s: continue
            for ch in _chunk_simple(s):
                if len(ch) >= MIN_CHARS:
                    chunks.append(ch); churn.append(u)
        if not chunks:
            continue

        # encode -> numpy float32
        vecs_out = []
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(chunks), ENC_BATCH):
                sub = chunks[i:i+ENC_BATCH]
                v = MODEL.encode(
                    sub,
                    batch_size=ENC_BATCH,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                vecs_out.append(np.asarray(v, dtype=np.float32))
        vecs = np.vstack(vecs_out) if len(vecs_out) > 1 else vecs_out[0]
        enc_dt = (time.time() - t0)

        out_q.put((vecs, chunks, churn))

        if time.time() - last_log >= HEARTBEAT_EVERY:
            r = len(chunks) / max(enc_dt, 1e-6)
            log(f"Worker-{wid}: {len(chunks)} chunks, enc={enc_dt*1000:.0f}ms (~{r:.1f} vec/s)",
                f"W{wid}", Fore.MAGENTA)
            last_log = time.time()

    out_q.put(None)
    log(f"Worker-{wid} tamamlandı", f"W{wid}", Fore.MAGENTA)

# ---- WRITER ----
def writer(in_q: Queue, stop_event: Event):
    # writer tek thread
    try: faiss.omp_set_num_threads(1)
    except Exception: pass
    torch.set_num_threads(1)

    store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)
    nt0 = store.index.ntotal if store.index else 0
    log(f"Writer başladı | ntotal={nt0}", "WRITER", Fore.GREEN)

    finished = 0
    added = 0
    last_hb = time.time()

    ACC_V, ACC_T, ACC_U = [], [], []

    def flush_add():
        nonlocal ACC_V, ACC_T, ACC_U, added
        if not ACC_V: return
        vecs = np.vstack(ACC_V).astype(np.float32, copy=False)
        dim  = vecs.shape[1]
        with store._lock:
            store._ensure_index(dim)
            faiss.normalize_L2(vecs)
            start = store._next_int_id
            ids = np.arange(start, start + vecs.shape[0], dtype=np.int64)
            store._next_int_id += vecs.shape[0]
            store.index.add_with_ids(vecs, ids)
            # hafif meta
            for j, fid in enumerate(ids):
                store.meta[int(fid)] = {
                    "external_id": os.urandom(8).hex(),
                    "metadata": {
                        "doc_type": DOC_TYPE,
                        "url": ACC_U[j],
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            # anlık kaydet (güvenli)
            store._save_state()
        added += vecs.shape[0]
        ACC_V.clear(); ACC_T.clear(); ACC_U.clear()
        log(f"Writer: +{vecs.shape[0]} (toplam={added})", "WRITER", Fore.GREEN)

    while finished < N_WORKERS and not stop_event.is_set():
        # heartbeat
        if time.time() - last_hb >= HEARTBEAT_EVERY:
            nt = store.index.ntotal if store.index else 0
            ms = os.path.getsize(META_PATH) if os.path.exists(META_PATH) else 0
            isz = os.path.getsize(INDEX_PATH) if os.path.exists(INDEX_PATH) else 0
            log(f"[HB] added={added} ntotal={nt} meta={fmt_bytes(ms)} index={fmt_bytes(isz)}",
                "WRITER", Fore.CYAN)
            last_hb = time.time()

        try:
            item = in_q.get(timeout=1.0)
        except queue.Empty:
            if sum(v.shape[0] for v in ACC_V) >= FLUSH_ADD_EVERY:
                flush_add()
            continue

        if item == WRITER_POISON:
            break
        if item is None:
            finished += 1
            continue

        vecs, texts, urls = item
        ACC_V.append(np.asarray(vecs, dtype=np.float32))
        ACC_T.extend(texts)
        ACC_U.extend(urls)

        if sum(v.shape[0] for v in ACC_V) >= FLUSH_ADD_EVERY:
            flush_add()

    # kalanlar
    flush_add()

    # final atomic
    try:
        with store._lock:
            flush_atomic(store)
    except Exception as e:
        log(f"final flush hata: {e}", "WRITER", Fore.YELLOW)

    nt = store.index.ntotal if store.index else 0
    log(f"Writer bitti | eklenen={added} | ntotal={nt}", "WRITER", Fore.GREEN)

# ---- MAIN ----
def main():
    try:
        set_start_method("fork", force=True)
    except RuntimeError:
        pass

    log(f"Sistem: {CPUS} CPU | workers={N_WORKERS}", "SYS", Fore.CYAN)
    preload_model()

    ensure_processed_file_exists()
    files = list_parquets(ROOT_DIR)
    if not files:
        log(f"Parquet bulunamadı: {ROOT_DIR}", "SYS", Fore.RED)
        sys.exit(1)

    stop_event = Event()
    q_r2w  = Queue(maxsize=Q_R2W_SIZE)
    q_w2wr = Queue(maxsize=Q_W2WR_SIZE)

    p_reader = Process(target=reader, args=(files, q_r2w, stop_event), daemon=True)
    p_writer = Process(target=writer, args=(q_w2wr, stop_event), daemon=False)
    workers  = [Process(target=worker, args=(q_r2w, q_w2wr, stop_event, i+1), daemon=True)
                for i in range(N_WORKERS)]

    def shutdown(*_):
        log("Kapanış sinyali alındı", "SYS", Fore.YELLOW)
        stop_event.set()
        try: q_w2wr.put_nowait(WRITER_POISON)
        except Exception: pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    atexit.register(lambda: stop_event.set())

    t0 = time.time()
    p_writer.start()
    for p in workers: p.start()
    p_reader.start()

    try:
        p_reader.join()
        for p in workers: p.join()
        try: q_w2wr.put(WRITER_POISON, timeout=2)
        except Exception: pass
        p_writer.join()
    finally:
        for p in workers:
            if p.is_alive(): p.terminate()
        if p_writer.is_alive(): p_writer.terminate()
        log(f"Toplam süre: {time.time()-t0:.1f}s", "SYS", Fore.CYAN)

if __name__ == "__main__":
    main()