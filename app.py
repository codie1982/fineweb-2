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
INDEX_PATH = os.path.abspath("faiss.index")
META_PATH  = os.path.abspath("meta.json")     # kullanılmayacak ama dursun
META_JSONL = os.path.abspath("meta.jsonl")    # <-- yeni
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

Q_R2W_SIZE  = N_WORKERS * 50
Q_W2WR_SIZE = N_WORKERS * 50

READ_DISPATCH = 2048        # reader->worker paket büyüklüğü
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150
ENC_BATCH     = 128         # CPU: 128–512 deneyebilirsin

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

        out_q.put((vecs, None))

        if time.time() - last_log >= HEARTBEAT_EVERY:
            r = len(chunks) / max(enc_dt, 1e-6)
            log(f"Worker-{wid}: {len(chunks)} chunks, enc={enc_dt*1000:.0f}ms (~{r:.1f} vec/s)",
                f"W{wid}", Fore.MAGENTA)
            last_log = time.time()

    out_q.put(None)
    log(f"Worker-{wid} tamamlandı", f"W{wid}", Fore.MAGENTA)

# ---- WRITER ----
def writer(in_q: Queue, stop_event: Event):
    try:
        try: faiss.omp_set_num_threads(1)
        except: pass
        try: torch.set_num_threads(1)
        except: pass

        # FAISS index: sadece index’i tutacağız (meta’yı RAM’de tutmuyoruz)
        store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)
        log(f"Writer başladı | CWD={os.getcwd()} | INDEX={INDEX_PATH} | JSONL={META_JSONL}", "WRITER", Fore.GREEN)

        # JSONL dosyasını append modunda aç (satır başına bir meta)
        os.makedirs(os.path.dirname(META_JSONL) or ".", exist_ok=True)
        meta_f = open(META_JSONL, "a", encoding="utf-8", buffering=1)  # satır-bazlı, line-buffered

        finished = 0
        added = 0
        last_hb = time.time()
        last_ckpt = time.time()

        ACC_V = []
        ACC_U = []                # urls
        FLUSH_ADD_EVERY = 2048
        CKPT_EVERY_SEC  = 10      # her 10 sn’de index checkpoint

        def sizes(prefix=""):
            try:
                idx = os.path.getsize(INDEX_PATH) if os.path.exists(INDEX_PATH) else 0
                jsl = os.path.getsize(META_JSONL) if os.path.exists(META_JSONL) else 0
                log(f"{prefix}index={idx/1024/1024:.2f}MB jsonl={jsl/1024/1024:.2f}MB", "WRITER", Fore.CYAN)
            except Exception as e:
                log(f"Boyut ölçüm hatası: {e}", "WRITER", Fore.YELLOW)

        def checkpoint_index():
            # sadece FAISS’i yaz; meta’yı JSONL’ye zaten yazıyoruz
            if store.index is None:
                return
            tmp = INDEX_PATH + ".tmp"
            faiss.write_index(store.index, tmp)
            os.replace(tmp, INDEX_PATH)

        def flush_add(force=False):
            nonlocal ACC_V, ACC_U, added
            n_acc = sum(v.shape[0] for v in ACC_V)
            if n_acc == 0:
                return
            if not force and n_acc < FLUSH_ADD_EVERY:
                return

            try:
                vecs = np.vstack(ACC_V).astype(np.float32, copy=False)
            except ValueError as e:
                log(f"❌ vstack hata: {e} | shapes={[v.shape for v in ACC_V]}", "WRITER", Fore.RED)
                ACC_V.clear(); ACC_U.clear()
                return

            dim = vecs.shape[1]
            n   = vecs.shape[0]
            with store._lock:
                store._ensure_index(dim)
                faiss.normalize_L2(vecs)

                start = store._next_int_id
                ids   = np.arange(start, start + n, dtype=np.int64)
                store._next_int_id += n

                store.index.add_with_ids(vecs, ids)

                # JSONL: her id/url için tek satır yaz
                now_str = time.strftime("%Y-%m-%d %H:%M:%S")
                for j, fid in enumerate(ids):
                    rec = {
                        "id": int(fid),
                        "url": ACC_U[j],
                        "doc_type": DOC_TYPE,
                        "created_at": now_str
                    }
                    meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                meta_f.flush()              # OS buffer
                os.fsync(meta_f.fileno())   # disk’e bastır

            added += n
            ACC_V.clear(); ACC_U.clear()
            log(f"Writer: +{n} (toplam={added})", "WRITER", Fore.GREEN)
            sizes("   ↳ ")

        while finished < N_WORKERS and not stop_event.is_set():
            now = time.time()

            # periyodik index checkpoint
            if now - last_ckpt >= CKPT_EVERY_SEC:
                checkpoint_index()
                last_ckpt = now

            # heartbeat
            if now - last_hb >= 30:
                nt = store.index.ntotal if store.index else 0
                log(f"[HB] added={added} ntotal={nt}", "WRITER", Fore.CYAN)
                sizes("   ↳ ")
                last_hb = now

            try:
                item = in_q.get(timeout=1.0)
            except queue.Empty:
                # kuyruk boşsa da ara ara yaz
                flush_add(force=True)
                continue

            if item == WRITER_POISON:
                log("Writer: poison", "WRITER", Fore.YELLOW)
                break

            if item is None:
                finished += 1
                continue

            # <<< değişti: worker sadece (vecs, urls) gönderiyor
            vecs, urls = item
            if urls is None:
                urls = [None] * vecs.shape[0]
            if not isinstance(vecs, np.ndarray) or vecs.size == 0:
                continue

            ACC_V.append(np.asarray(vecs, dtype=np.float32))
            ACC_U.extend(urls)

            if sum(v.shape[0] for v in ACC_V) >= FLUSH_ADD_EVERY:
                flush_add(force=True)

        # kalanlar
        flush_add(force=True)
        checkpoint_index()

        # kapat
        try:
            meta_f.flush()
            os.fsync(meta_f.fileno())
            meta_f.close()
        except Exception:
            pass

        nt = store.index.ntotal if store.index else 0
        log(f"✅ Writer bitti | eklenen={added} | ntotal={nt}", "WRITER", Fore.GREEN)
        sizes("Final: ")

    except Exception as e:
        log(f"❌ Writer hatası: {e}", "WRITER", Fore.RED)
        import traceback; traceback.print_exc()
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