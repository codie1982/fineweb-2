# -*- coding: utf-8 -*-
"""
Sade ve sağlam ingest (CPU-only)
Reader -> Worker -> Writer veri hattı
- Reader: parquet'ten (text,url) paketleri
- Worker: chunk + batch encode -> (vecs, urls)
- Writer: toplu FAISS add + periyodik checkpoint + JSONL meta
"""

import os, glob, json, time, sys, signal, atexit, queue
from typing import List, Optional, Set
from multiprocessing import Process, Queue, Event, set_start_method, cpu_count

import numpy as np
import pyarrow.dataset as ds
import torch
import faiss
from sentence_transformers import SentenceTransformer

# --- basit log ---
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True); COLOR=True
except Exception:
    COLOR=False
    class _D:  # fallback
        def __getattr__(self, k): return ""
    Fore=Style=_D()

def log(msg, tag="SYS", color=Fore.CYAN):
    line=f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    if COLOR: print(f"{color}{line}{Style.RESET_ALL}", flush=True)
    else: print(line, flush=True)

# --- ayarlar ---
ROOT_DIR = os.path.abspath("./fineweb-2/data/tur_Latn/train")
log(f"ROOT_DIR: {ROOT_DIR} | exists={os.path.exists(ROOT_DIR)}", "SYS")
INDEX_PATH = os.path.abspath("faiss.index")
META_JSONL = os.path.abspath("meta.jsonl")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CPUS       = cpu_count()
N_WORKERS  = max(4, min(12, CPUS - 2))     # 2 core reader+writer
LIMIT_TOTAL = None                          # tüm veri için None

BATCH_READ      = 4096
MIN_CHARS       = 80
MAX_CHARS       = 8000
READ_DISPATCH   = 512                       # küçük paketler = düşük RAM
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 100
ENC_BATCH       = 64                        # CPU için güvenli
FLUSH_ADD_EVERY = 256                       # writer sık flush
CHECKPOINT_EVERY_S = 20

TEXT_CANDS = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS  = ["url","source_url","link","origin","canonical_url"]

Q_R2W_SIZE = N_WORKERS * 20
Q_W2WR_SIZE = N_WORKERS * 20

WRITER_POISON = "__WRITER_POISON__"

# --- yardımcılar ---
def list_parquets(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))

def pick_col(schema_names: List[str], cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in schema_names: return c
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
    s=["B","KB","MB","GB","TB"]; i=0; x=float(n)
    while x>=1024 and i<len(s)-1: x/=1024; i+=1
    return f"{x:.1f}{s[i]}"

# --- global model (fork öncesi) ---
MODEL: Optional[SentenceTransformer] = None
def preload_model():
    global MODEL
    if MODEL is not None: return MODEL
    try: torch.set_num_threads(1)  # her worker tek thread
    except Exception: pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = SentenceTransformer(MODEL_NAME, device=device); MODEL.eval()
    log(f"Encoder hazır: {MODEL_NAME} | device={device} | workers={N_WORKERS}")
    return MODEL

# --- reader ---
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
        out_q.put((bufT, bufU))  # tek put = paket
        total += len(bufT)
        bufT, bufU = [], []

    sent_limit = LIMIT_TOTAL if isinstance(LIMIT_TOTAL, int) else 10**15

    for pidx, path in enumerate(todo):
        if stop_event.is_set() or total >= sent_limit: break
        log(f"[{pidx+1}/{len(todo)}] read: {os.path.basename(path)}", "READER", Fore.BLUE)

        try:
            ds_ = ds.dataset(path, format="parquet")
        except Exception as e:
            log(f"❌ okunamadı: {path} -> {e}", "READER", Fore.RED)
            mark_processed(path); continue

        names = ds_.schema.names
        tcol  = pick_col(names, TEXT_CANDS); ucol = pick_col(names, URL_CANDS)
        if not tcol:
            log(f"⚠️ metin kolonu yok, atla: {path}", "READER", Fore.YELLOW)
            mark_processed(path); continue

        scanner = ds_.scanner(columns=[tcol] + ([ucol] if ucol else []),
                              batch_size=BATCH_READ, use_threads=True)

        for b in scanner.to_batches():
            if stop_event.is_set() or total >= sent_limit: break
            d = b.to_pydict()
            texts = d.get(tcol, [])
            urls  = d.get(ucol, []) if ucol else [None]*len(texts)
            for i, t in enumerate(texts):
                if stop_event.is_set() or total >= sent_limit: break
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

# --- worker ---
def _chunk_simple(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    n=len(s); step=max(1,size-overlap)
    for st in range(0,n,step):
        ed=min(n, st+size)
        yield s[st:ed]

def worker(in_q: Queue, out_q: Queue, stop_event: Event, wid: int):
    try:
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
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

        texts, urls = item  # protokol: reader -> (text list, url list)

        chunks, ulist = [], []
        for s,u in zip(texts, urls):
            if not s: continue
            s = s.strip()
            if not s: continue
            for ch in _chunk_simple(s):
                if len(ch) >= MIN_CHARS:
                    chunks.append(ch); ulist.append(u)
        if not chunks: continue

        vecs_out=[]; t0=time.time()
        try:
            with torch.inference_mode():
                for i in range(0,len(chunks),ENC_BATCH):
                    sub = chunks[i:i+ENC_BATCH]
                    v = MODEL.encode(sub, batch_size=ENC_BATCH,
                                     show_progress_bar=False,
                                     normalize_embeddings=True,
                                     convert_to_numpy=True)
                    vecs_out.append(np.asarray(v, dtype=np.float32))
        except Exception as e:
            log(f"❌ encode hata: {e}", f"W{wid}", Fore.RED)
            continue
        vecs = np.vstack(vecs_out) if len(vecs_out)>1 else vecs_out[0]

        # protokol: worker -> writer : (vecs, urls)
        out_q.put((vecs, ulist))

        if time.time()-last_log >= 30:
            enc_s = time.time()-t0
            rate = len(chunks)/max(enc_s,1e-6)
            log(f"Worker-{wid}: chunks={len(chunks)} enc={enc_s*1000:.0f}ms (~{rate:.1f} vec/s)",
                f"W{wid}", Fore.MAGENTA)
            last_log=time.time()

    out_q.put(None)
    log(f"Worker-{wid} tamamlandı", f"W{wid}", Fore.MAGENTA)

# --- writer ---
def writer(in_q: Queue, stop_event: Event):
    try:
        faiss.omp_set_num_threads(1)
        torch.set_num_threads(1)
    except Exception:
        pass

    # Klasörler
    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(META_JSONL) or ".", exist_ok=True)

    # Hata yakalayıcı blok
    try:
        # JSONL dosyasını kesin oluştur (touch)
        with open(META_JSONL, "a", encoding="utf-8") as _touch:
            pass

        # Boş bir indexi de daha en başta yaz (writer’ın başladığını kanıtlar)
        # (Dim bilinmediği için geçici 1-dimlik boş index yazıyoruz; ilk gerçek add'te üzerine yazacağız.)
        tmp_init = INDEX_PATH + ".init"
        faiss.write_index(faiss.IndexFlatIP(1), tmp_init)
        os.replace(tmp_init, INDEX_PATH)

        log(f"Writer başladı | INDEX={INDEX_PATH} | JSONL={META_JSONL}", "WRITER", Fore.GREEN)
    except Exception as e:
        log(f"❌ Writer init hata: {e}", "WRITER", Fore.RED)
        # Writer devam etmesin; ana süreç temiz kapatır
        return

# --- main ---
def main():
    try: set_start_method("fork", force=True)
    except RuntimeError: pass

    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(META_JSONL) or ".", exist_ok=True)

    log(f"Sistem: {cpu_count()} CPU | workers={N_WORKERS}", "SYS", Fore.CYAN)
    preload_model()

    ensure_processed_file_exists()
    files = list_parquets(ROOT_DIR)
    if not files:
        log(f"❌ Parquet bulunamadı: {ROOT_DIR}", "SYS", Fore.RED); sys.exit(1)

    stop_event = Event()
    q_r2w  = Queue(maxsize=Q_R2W_SIZE)
    q_w2wr = Queue(maxsize=Q_W2WR_SIZE)

    p_reader = Process(target=reader, args=(files, q_r2w, stop_event), daemon=True)
    p_writer = Process(target=writer, args=(q_w2wr, stop_event), daemon=False)
    workers  = [Process(target=worker, args=(q_r2w, q_w2wr, stop_event, i+1), daemon=False)
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