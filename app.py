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

# --- colorama (opsiyonel, yoksa sade çıktı) ---

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
ROOT_DIR         = "./fineweb-2/data/tur_Latn/train"    # klasörü istediğin gibi ayarla
INDEX_PATH       = "faiss.index"
META_PATH        = "meta.json"
MODEL_NAME       = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

N_WORKERS        = 12
LIMIT_TOTAL      = 100          # tamamını işle (kısa test için 100 gibi ver)
BATCH_READ       = 4096
BATCH_ENCODE     = 24
MIN_CHARS        = 80
MAX_CHARS        = 8000
DOC_TYPE         = "fineweb2"
MAX_TEXT_CHARS   = 1500          # HIZ için metni kırp (None yaparsan kırpmaz)
MAX_SEG_LENGTH   = 128
TEXT_CANDS       = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS        = ["url","source_url","link","origin","canonical_url"]

Q_R2W_SIZE       = 2000          # büyük kuyruklar → daha akıcı pipeline
Q_W2WR_SIZE      = 2000

FLUSH_EVERY      = 2000          # writer checkpoint (vektör)
FLUSH_INTERVAL_S = 30            # writer checkpoint (saniye)

PROCESSED_FILE   = "processed_files.txt"  # dosya-bazlı resume için
LOG_FILE         = None           # "logs/ingest.log" yazarsan dosyaya da loglar
# =====================================

def processed_file_path() -> str:
    # ► mutlak yol (görebilmen için loga basacağız)
    return os.path.abspath(PROCESSED_FILE)

def ensure_processed_file_exists():
    path = processed_file_path()
    if not os.path.exists(path):
        # dosya yarat (atomic değil ama tek yazar 'reader' olduğu için yeterli)
        with open(path, "w", encoding="utf-8") as f:
            f.write("")  # boş oluştur
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
    # ► hemen diske (flush + fsync)
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
    """faiss.index ve meta.json'u temp dosyaya yazıp atomik rename ile değiştir."""
    meta_tmp = store.meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in store.meta.items()}, f, ensure_ascii=False)
    os.replace(meta_tmp, store.meta_path)

    idx_tmp = store.index_path + ".tmp"
    faiss.write_index(store.index, idx_tmp)
    os.replace(idx_tmp, store.index_path)

# ----- Global model (fork paylaşımı) -----
MODEL: Optional[SentenceTransformer] = None
def preload_model():
    global MODEL
    if MODEL is None:
        log("Model yükleniyor… (tek sefer)", "SYS", Fore.CYAN)
        MODEL = SentenceTransformer(MODEL_NAME)
        MODEL.max_seq_length = MAX_SEG_LENGTH
        MODEL.eval()
        log("Model hazır.", "SYS", Fore.CYAN)

# ----- processed_files resume -----




# -------------- PROCESSES --------------
def reader(files: List[str], out_q: Queue, stop_event: Event):
    """
    Parquet dosyalarını tarayıp (text,url) örneklerini kuyruğa gönderir.
    processed_files.txt’de olanları atlar (resume).
    """
    sent = 0
    total_limit = LIMIT_TOTAL if isinstance(LIMIT_TOTAL, int) else float("inf")
    seen_files = load_processed()
    todo = [p for p in files if p not in seen_files]
    log(f"Reader başladı: {len(files)} dosya (atlanan={len(seen_files)}, işlenecek={len(todo)}). Limit={total_limit if total_limit!=float('inf') else 'ALL'}", "READER", Fore.BLUE)

    for path in todo:
        if stop_event.is_set() or sent >= total_limit:
            break
        try:
            ds_ = ds.dataset(path, format="parquet")
        except Exception as e:
            log(f"❌ okunamadı: {path} -> {e}", "READER", Fore.RED)
            # bozuksa yine processed olarak işaretleyelim ki takılmasın (istersen etme)
            mark_processed(path)
            continue

        names = ds_.schema.names
        tcol  = pick_col(names, TEXT_CANDS)
        ucol  = pick_col(names, URL_CANDS)
        if not tcol:
            log(f"⚠️ metin kolonu yok, atla: {path} -> {names}", "READER", Fore.YELLOW)
            mark_processed(path)
            continue

        scanner = ds_.scanner(columns=[tcol] + ([ucol] if ucol else []), batch_size=BATCH_READ,use_threads=True)
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
                out_q.put((s, u))  # backpressure normal
                sent += 1
                if sent % 5000 == 0:
                    log(f"[READER] queued={sent}", "READER", Fore.BLUE)
            if sent >= total_limit: break

        # bu dosya bitti — processed olarak işaretle
        mark_processed(path)

    # worker'lara bitiş sinyali
    for _ in range(N_WORKERS):
        out_q.put(None)
    log(f"✅ Reader bitti. Kuyruğa {min(sent, total_limit)} örnek gönderildi.", "READER", Fore.BLUE)


def worker(in_q, out_q, stop_event, wid: int):
    import gc, time
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
    PARTIAL_FLUSH_S = 10      # 🔸 batch 24’e ulaşmasa bile 10 sn dolunca gönder

    while True:
        if stop_event.is_set():
            # elde kalan varsa gönder
            if batch_texts:
                vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE,
                                    show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
                out_q.put((vecs, batch_texts, batch_urls))
                processed += len(batch_texts)
                del vecs; gc.collect()
            out_q.put(None)
            break

        try:
            item = in_q.get(timeout=2)
        except queue.Empty:
            # 🔸 zaman bazlı kısmi flush
            if batch_texts and (time.time() - last_push_t) >= PARTIAL_FLUSH_S:
                vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE,
                                    show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
                out_q.put((vecs, batch_texts, batch_urls))
                processed += len(batch_texts)
                batch_texts, batch_urls = [], []
                del vecs; gc.collect()
                last_push_t = time.time()
            continue

        if item is None:
            # reader bitti → elde kalan varsa gönder ve çık
            if batch_texts:
                vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE,
                                    show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
                out_q.put((vecs, batch_texts, batch_urls))
                processed += len(batch_texts)
                del vecs; gc.collect()
            out_q.put(None)
            break

        text, url = item
        batch_texts.append(text)
        batch_urls.append(url)

        # 🔸 geleni sayıp ara ara logla (tüketim var mı görelim)
        if processed % 1000 == 0 and processed > 0:
            log(f"Worker-{wid} tüketim ilerleme: {processed}", f"W{wid}", Fore.MAGENTA)

        # 🔸 batch dolduysa gönder
        if len(batch_texts) >= BATCH_ENCODE:
            vecs = MODEL.encode(batch_texts, batch_size=BATCH_ENCODE,
                                show_progress_bar=False, normalize_embeddings=True).astype(np.float16)
            out_q.put((vecs, batch_texts, batch_urls))
            processed += len(batch_texts)
            batch_texts, batch_urls = [], []
            del vecs; gc.collect()
            last_push_t = time.time()

    log(f"Worker-{wid} tamamlandı. İşlenen: {processed}", f"W{wid}", Fore.MAGENTA)



def writer(in_q: Queue, stop_event: Event):
    """Writer: FAISS + meta'ya toplu ekler, periyodik atomik checkpoint + dup guard + heartbeat."""
    try:
        store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)

        seen_sha1 = set()
        for v in store.meta.values():
            h = (v.get("metadata") or {}).get("sha1")
            if h: seen_sha1.add(h)

        finished = 0
        total_added = 0
        t0 = time.time()
        last_flush_n = 0
        last_flush_t = time.time()

        received_total = 0
        skipped_total  = 0
        kept_total     = 0
        last_hb_t      = time.time()

        log("Writer başladı.", "WRITER", Fore.GREEN)

        def heartbeat():
            nonlocal last_hb_t
            now = time.time()
            if now - last_hb_t >= 60:
                log(f"[HB] recv={received_total} kept={kept_total} skipped={skipped_total} ntotal={store.index.ntotal if store.index else 0}", "WRITER", Fore.GREEN)
                last_hb_t = now

        def maybe_flush():
            nonlocal last_flush_n, last_flush_t
            now = time.time()
            if (total_added - last_flush_n) >= FLUSH_EVERY or (now - last_flush_t) >= FLUSH_INTERVAL_S:
                try:
                    flush_atomic(store)
                    last_flush_n = total_added
                    last_flush_t = now
                    log(f"💾 Checkpoint: ntotal={store.index.ntotal if store.index else 0}", "WRITER", Fore.GREEN)
                except Exception as e:
                    log(f"⚠️ Flush hatası: {e}", "WRITER", Fore.YELLOW)

        while True:
            # ÇIKIŞ KOŞULU: tüm worker'lardan 'None' alındıysa bitir
            if finished >= N_WORKERS:
                break

            try:
                item = in_q.get(timeout=5)
            except queue.Empty:
                heartbeat()
                maybe_flush()
                # finished sayısı artmışsa bir sonraki turda break olacak
                continue

            if item is None:
                finished += 1
                # burada doğrudan continue; üstteki başta kontrol kıracak
                continue

            vecs, texts, urls = item
            vecs = np.asarray(vecs)
            if vecs.dtype == np.float16:
                vecs = vecs.astype(np.float32)

            received_total += len(texts)

            keep_idx = []
            sha1_list = []
            local_skipped = 0
            for i, txt in enumerate(texts):
                h = text_sha1(txt)
                if h in seen_sha1:
                    local_skipped += 1
                    continue
                keep_idx.append(i)
                sha1_list.append(h)
            skipped_total += local_skipped

            if not keep_idx:
                heartbeat()
                maybe_flush()
                continue

            vecs = np.asarray(vecs, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            vecs = vecs[keep_idx, :]
            texts = [texts[i] for i in keep_idx]
            urls  = [urls[i]  for i in keep_idx]
            kept_total += len(texts)

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
                        "metadata": {
                            "doc_type": DOC_TYPE,
                            "url": urls[j],
                            "h_path": ["# Loose"],
                            "sha1": sha1_list[j],
                        },
                    }
                    seen_sha1.add(sha1_list[j])

            total_added += len(texts)
            if total_added % 5000 == 0 and total_added > 0:
                dt = time.time() - t0
                rps = total_added / max(dt, 1e-6)
                log(f"[RATE] added={total_added} | {rps:.1f} vec/s", "WRITER", Fore.GREEN)

            heartbeat()
            maybe_flush()

        # son flush (atomik)
        try:
            flush_atomic(store)
        except Exception as e:
            log(f"⚠️ Final flush hatası: {e}", "WRITER", Fore.YELLOW)

        log(f"✅ Writer tamamlandı. Toplam {total_added} vektör | {time.time()-t0:.1f}s", "WRITER", Fore.GREEN)
    except Exception as e:
        log(f"⚠️ Writer hatası: {e}", "WRITER", Fore.YELLOW)

# -------------- MAIN --------------
def main():
    try: set_start_method("fork", force=True)
    except RuntimeError: pass

    # …
    pf = ensure_processed_file_exists()
    log(f"Resume dosyası: {pf}", "SYS", Fore.CYAN)

    # thread sınırları
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    preload_model()

    files = list_parquets(ROOT_DIR)
    if not files:
        log(f"❌ Parquet bulunamadı: {ROOT_DIR}", "SYS", Fore.RED); sys.exit(1)
    info_limit = LIMIT_TOTAL if LIMIT_TOTAL is not None else "ALL"
    log(f"{len(files)} dosya | workers={N_WORKERS} | limit={info_limit}", "SYS", Fore.CYAN)

    stop_event = Event()

    q_r2w  = Queue(maxsize=Q_R2W_SIZE)
    q_w2wr = Queue(maxsize=Q_W2WR_SIZE)

    p_reader = Process(target=reader, args=(files, q_r2w, stop_event), daemon=True)
    p_writer = Process(target=writer, args=(q_w2wr, stop_event), daemon=True)
    workers  = [Process(target=worker, args=(q_r2w, q_w2wr, stop_event, i+1), daemon=True) for i in range(N_WORKERS)]

    def graceful_shutdown(signum=None, frame=None):
        log(f"⚠️ Sinyal alındı ({signum}). Kapanış başlatılıyor…", "SYS", Fore.YELLOW)
        stop_event.set()
        try:
            for _ in range(N_WORKERS):
                q_r2w.put_nowait(None)
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
        p_writer.join()
    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)
        for p in workers:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
        p_writer.join(timeout=10)
        if p_writer.is_alive(): p_writer.terminate()
    finally:
        log(f"🏁 Toplam süre: {time.time()-t0:.1f}s", "SYS", Fore.CYAN)
        log(f"FAISS: {INDEX_PATH} | META: {META_PATH}", "SYS", Fore.CYAN)

if __name__ == "__main__":
    main()
