# -*- coding: utf-8 -*-
import os, glob, queue, time, datetime
from typing import List, Optional
from multiprocessing import Process, Queue
import numpy as np
import pyarrow.dataset as ds
import faiss
import torch
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style, init
from embedding_index import EmbeddingIndex

init(autoreset=True)

# ==== AYARLAR ====
ROOT_DIR      = "./fineweb-2/data/tur_Latn/train"
INDEX_PATH    = "faiss.index"
META_PATH     = "meta.json"
MODEL_NAME    = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

LIMIT_TOTAL   = 1000        # kısa test
BATCH_READ    = 2048
BATCH_ENCODE  = 32
MIN_CHARS     = 80
MAX_CHARS     = 8000
DOC_TYPE      = "fineweb2"
N_WORKERS     = 12

TEXT_CANDS    = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS     = ["url","source_url","link","origin","canonical_url"]
# ==================

def log(msg, tag="SYS", color=Fore.CYAN):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{now}] [{tag}] {msg}{Style.RESET_ALL}")

def list_parquets(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))

def pick_col(schema_names: List[str], cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in schema_names: return c
    return None

# ---------- Reader ----------
def reader(files: List[str], out_q: Queue):
    sent = 0
    log(f"Başladı ({len(files)} dosya).", "READER", Fore.BLUE)
    for path in files:
        if sent >= LIMIT_TOTAL: break
        try:
            ds_ = ds.dataset(path, format="parquet")
        except Exception as e:
            log(f"❌ okunamadı: {path} -> {e}", "READER", Fore.RED)
            continue

        names = ds_.schema.names
        tcol  = pick_col(names, TEXT_CANDS)
        ucol  = pick_col(names, URL_CANDS)
        if not tcol:
            log(f"⚠️ metin kolonu yok, atla: {path}", "READER", Fore.YELLOW)
            continue

        scanner = ds_.scanner(columns=[tcol] + ([ucol] if ucol else []), batch_size=BATCH_READ)
        for b in scanner.to_batches():
            d = b.to_pydict()
            texts = d.get(tcol, [])
            urls  = d.get(ucol, []) if ucol else [None]*len(texts)
            for i, t in enumerate(texts):
                if sent >= LIMIT_TOTAL: break
                if not t: continue
                s = str(t).strip()
                if not (MIN_CHARS <= len(s) <= MAX_CHARS): continue
                u = urls[i] if urls and urls[i] else f"http://fw2.local/{os.path.basename(path)}-{i}"
                out_q.put((s, u))
                sent += 1
            if sent >= LIMIT_TOTAL: break
    for _ in range(N_WORKERS): out_q.put(None)
    log(f"✅ {sent} satır okundu ve kuyruğa gönderildi.", "READER", Fore.BLUE)

# ---------- Worker ----------
def worker(in_q: Queue, out_q: Queue, wid: int):
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    model = SentenceTransformer(MODEL_NAME)
    model.eval()

    log(f"Worker-{wid} başladı.", f"W{wid}", Fore.MAGENTA)
    processed = 0
    batch_texts, batch_urls = [], []

    while True:
        try:
            item = in_q.get(timeout=10)
        except queue.Empty:
            continue

        if item is None:
            if batch_texts:
                vecs = model.encode(batch_texts, batch_size=BATCH_ENCODE, normalize_embeddings=True).astype(np.float32)
                out_q.put((vecs, batch_texts, batch_urls))
            out_q.put(None)
            log(f"Worker-{wid} tamamlandı ({processed} kayıt).", f"W{wid}", Fore.MAGENTA)
            break

        text, url = item
        batch_texts.append(text)
        batch_urls.append(url)
        processed += 1

        if len(batch_texts) >= BATCH_ENCODE:
            vecs = model.encode(batch_texts, batch_size=BATCH_ENCODE, normalize_embeddings=True).astype(np.float32)
            out_q.put((vecs, batch_texts, batch_urls))
            log(f"Worker-{wid}: {processed} kayıt işlendi.", f"W{wid}", Fore.MAGENTA)
            batch_texts, batch_urls = [], []

# ---------- Writer ----------
def writer(in_q: Queue):
    store = EmbeddingIndex(MODEL_NAME, INDEX_PATH, META_PATH)
    finished = 0
    total_added = 0
    t0 = time.time()
    log("Writer başladı.", "WRITER", Fore.GREEN)

    while True:
        try:
            item = in_q.get(timeout=20)
        except queue.Empty:
            if finished >= N_WORKERS:
                break
            continue

        if item is None:
            finished += 1
            if finished >= N_WORKERS:
                break
            continue

        vecs, texts, urls = item
        dim = vecs.shape[1]
        with store._lock:
            store._ensure_index(dim)
            start = store._next_int_id
            ids = np.arange(start, start + len(texts), dtype=np.int64)
            store._next_int_id += len(texts)
            store.index.add_with_ids(vecs, ids)
            for i, fid in enumerate(ids):
                store.meta[int(fid)] = {
                    "external_id": os.urandom(8).hex(),
                    "text": texts[i],
                    "metadata": {"doc_type": DOC_TYPE, "url": urls[i]},
                }
        total_added += len(texts)
        if total_added % 20 == 0:
            log(f"💾 {total_added} vektör eklendi.", "WRITER", Fore.GREEN)

    store._save_state()
    elapsed = time.time() - t0
    log(f"✅ Writer tamamlandı. Toplam {total_added} vektör | {elapsed:.1f}s", "WRITER", Fore.GREEN)

# ---------- Main ----------
def main():
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
    })

    files = list_parquets(ROOT_DIR)
    if not files:
        log(f"❌ Parquet bulunamadı: {ROOT_DIR}", "SYS", Fore.RED)
        return

    log(f"{len(files)} dosya bulundu | {N_WORKERS} worker | limit={LIMIT_TOTAL}", "SYS", Fore.CYAN)

    q_r2w = Queue(maxsize=128)
    q_w2wr = Queue(maxsize=128)

    p_reader = Process(target=reader, args=(files, q_r2w), daemon=True)
    p_writer = Process(target=writer, args=(q_w2wr,), daemon=True)
    workers = [Process(target=worker, args=(q_r2w, q_w2wr, i+1), daemon=True) for i in range(N_WORKERS)]

    t0 = time.time()
    p_writer.start()
    for p in workers: p.start()
    p_reader.start()

    p_reader.join()
    for p in workers: p.join()
    p_writer.join()
    t1 = time.time()
    log(f"🏁 Toplam süre: {t1 - t0:.1f} sn", "SYS", Fore.CYAN)
    log(f"FAISS: {INDEX_PATH} | META: {META_PATH}", "SYS", Fore.CYAN)

if __name__ == "__main__":
    main()


