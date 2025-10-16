# -*- coding: utf-8 -*-
"""
Optimized multiprocessing ingest - CPU verimliliği artırıldı
"""

import os, glob, queue, time, datetime, sys, signal, atexit, json, hashlib
from typing import List, Optional, Set
from multiprocessing import Process, Queue, Event, set_start_method, cpu_count
import threading

import numpy as np
import pyarrow.dataset as ds
import faiss
import torch
from sentence_transformers import SentenceTransformer

# --- colorama ---
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except Exception:
    COLOR = False
    class _Dummy:
        def __getattr__(self, k): return ""
    Fore = Style = _Dummy()

from embedding_index import EmbeddingIndex

# ============== OPTIMIZED AYARLAR ==============
ROOT_DIR         = "./fineweb-2/data/tur_Latn/train"
INDEX_PATH       = "faiss.index"
META_PATH        = "meta.json"
MODEL_NAME       = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# CPU sayısına göre otomatik ayar
AVAILABLE_CPUS = cpu_count()
N_WORKERS        = min(AVAILABLE_CPUS - 2, 16)  # 2 core reader+writer için
if N_WORKERS < 4:
    N_WORKERS = 4

LIMIT_TOTAL      = 100
BATCH_READ       = 8192  # Artırıldı
MIN_CHARS        = 80
MAX_CHARS        = 8000
DOC_TYPE         = "fineweb2"
TEXT_CANDS       = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS        = ["url","source_url","link","origin","canonical_url"]
# Kuyruk boyutları optimize edildi
Q_R2W_SIZE       = N_WORKERS * 500  # Worker sayısına göre skalala
Q_W2WR_SIZE      = N_WORKERS * 300

LOG_FILE         = None
WRITER_POISON    = "__WRITER_POISON__"

# Throughput ayarları optimize edildi
READ_DISPATCH    = 8192   # Artırıldı
CHUNK_SIZE       = 1500
CHUNK_OVERLAP    = 200
ENC_BATCH        = 256    # Batch boyutu optimize edildi

# Yeni: Paralellik kontrolleri
MAX_QUEUE_WAIT   = 0.1    # Queue timeout (saniye)
FLUSH_INTERVAL   = 1000   # Writer flush interval
# =====================================

# Global Model
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
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    
    # Device detection optimize edildi
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True  # CUDA optimizasyonu
    else:
        device = "cpu"
        # CPU için thread ayarı
        torch.set_num_threads(min(4, AVAILABLE_CPUS // N_WORKERS))
    
    MODEL = SentenceTransformer(MODEL_NAME, device=device)
    MODEL.eval()
    
    # Mixed precision için hazırlık
    if device == "cuda":
        try:
            MODEL = MODEL.half()
            log(f"Model FP16 mode enabled", "SYS", Fore.CYAN)
        except Exception as e:
            log(f"FP16 not available: {e}", "SYS", Fore.YELLOW)
    
    log(f"Encoder hazır: {MODEL_NAME} | device={device} | workers={N_WORKERS}", "SYS", Fore.CYAN)
    return MODEL

# -------------- OPTIMIZED READER --------------
def reader(files: List[str], out_q: Queue, stop_event: Event):
    sent = 0
    total_limit = LIMIT_TOTAL if isinstance(LIMIT_TOTAL, int) else float("inf")
    seen_files = load_processed()
    todo = [p for p in files if p not in seen_files]
    log(f"Reader başladı: {len(files)} dosya (atlanan={len(seen_files)}, işlenecek={len(todo)}).", "READER", Fore.BLUE)

    batch_T, batch_U = [], []
    batch_count = 0
    
    def flush_reader_batch(force=False):
        nonlocal batch_T, batch_U, sent, batch_count
        if (batch_count > 0 and (force or batch_count >= READ_DISPATCH)):
            try:
                out_q.put((batch_T.copy(), batch_U.copy()), timeout=MAX_QUEUE_WAIT)
                sent += batch_count
                log(f"Reader: {batch_count} satır gönderildi (toplam: {sent})", "READER", Fore.BLUE)
            except queue.Full:
                log("⚠️ Reader queue dolu, bekleniyor...", "READER", Fore.YELLOW)
                time.sleep(0.1)
                return False
            
            batch_T.clear()
            batch_U.clear()
            batch_count = 0
        return True

    for path_idx, path in enumerate(todo):
        if stop_event.is_set() or sent >= total_limit:
            break
            
        log(f"[FILE {path_idx+1}/{len(todo)}] reading: {os.path.basename(path)}", "READER", Fore.BLUE)
        
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
        
        for batch_idx, b in enumerate(scanner.to_batches()):
            if stop_event.is_set() or sent >= total_limit: 
                break
                
            d = b.to_pydict()
            texts = d.get(tcol, [])
            urls  = d.get(ucol, []) if ucol else [None]*len(texts)
            
            for i, t in enumerate(texts):
                if stop_event.is_set() or sent >= total_limit: 
                    break
                if not t: 
                    continue
                    
                s = str(t).strip()
                if not (MIN_CHARS <= len(s) <= MAX_CHARS): 
                    continue
                    
                u = urls[i] if urls and urls[i] else f"http://fw2.local/{os.path.basename(path)}-{i}"

                batch_T.append(s)
                batch_U.append(u)
                batch_count += 1

                # Daha sık flush ile memory kullanımını optimize et
                if batch_count >= READ_DISPATCH:
                    if not flush_reader_batch(force=True):
                        # Queue doluysa biraz bekle
                        time.sleep(0.05)

        # Dosya bitişinde flush
        flush_reader_batch(force=True)
        mark_processed(path)

    # Son flush
    flush_reader_batch(force=True)
    
    # Worker'lara bitiş sinyali
    for _ in range(N_WORKERS):
        try:
            out_q.put(None, timeout=1.0)
        except queue.Full:
            log("⚠️ Bitiş sinyali gönderilemedi, queue dolu", "READER", Fore.YELLOW)
            
    log(f"✅ Reader bitti. Toplam {sent} satır gönderildi.", "READER", Fore.BLUE)

# -------------- OPTIMIZED WORKER --------------
def _chunk_simple(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    n = len(s)
    step = max(1, size - overlap)
    for start in range(0, n, step):
        end = min(n, start + size)
        yield s[start:end]

def worker(in_q, out_q, stop_event, wid: int):
    import gc
    
    # Thread ayarları optimize edildi
    try: 
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
    except Exception: 
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model loading optimize edildi
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME, device=device)
        MODEL.eval()
        if device == "cuda":
            try: 
                MODEL = MODEL.half()
            except Exception: 
                pass
    model = MODEL

    log(f"Worker-{wid} başladı. device={device}", f"W{wid}", Fore.MAGENTA)
    
    processed_count = 0
    last_log_time = time.time()
    
    while not stop_event.is_set():
        try:
            item = in_q.get(timeout=0.5)  # Daha kısa timeout
            if item is None:
                break
                
            texts, urls = item
            processed_count += len(texts)
            
            # Batch processing optimize edildi
            batch_texts, batch_urls = [], []
            for s, u in zip(texts, urls):
                if not s: 
                    continue
                s = s.strip()
                if not s: 
                    continue
                    
                for ch in _chunk_simple(s):
                    if len(ch) >= MIN_CHARS:
                        batch_texts.append(ch)
                        batch_urls.append(u)

            if not batch_texts:
                continue

            # Encode işlemi
            vecs_out = []
            t0 = time.time()
            
            with torch.inference_mode():
                for i in range(0, len(batch_texts), ENC_BATCH):
                    sub = batch_texts[i:i+ENC_BATCH]
                    v = model.encode(
                        sub,
                        batch_size=ENC_BATCH,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_tensor=True  # GPU için optimize
                    )
                    vecs_out.append(v.cpu().numpy() if device == "cuda" else np.asarray(v))

            vecs = np.vstack(vecs_out) if len(vecs_out) > 1 else vecs_out[0]
            dt = time.time() - t0

            # Writer'a gönder
            try:
                out_q.put((vecs, batch_texts, batch_urls), timeout=MAX_QUEUE_WAIT)
            except queue.Full:
                log(f"⚠️ Worker-{wid}: Writer queue dolu, bekleniyor...", f"W{wid}", Fore.YELLOW)
                time.sleep(0.1)
                # Tekrar dene
                try:
                    out_q.put((vecs, batch_texts, batch_urls), timeout=MAX_QUEUE_WAIT)
                except queue.Full:
                    log(f"❌ Worker-{wid}: Writer queue doldu, batch kaybedildi", f"W{wid}", Fore.RED)

            # Performans logging
            current_time = time.time()
            if current_time - last_log_time > 30:  # 30 saniyede bir log
                rate = processed_count / (current_time - last_log_time) if current_time > last_log_time else 0
                log(f"Worker-{wid}: {processed_count} docs | {rate:.1f} doc/s", f"W{wid}", Fore.MAGENTA)
                processed_count = 0
                last_log_time = current_time

            # Bellek optimizasyonu
            if len(vecs_out) > 1:
                del vecs_out
            gc.collect()

        except queue.Empty:
            continue
        except Exception as e:
            log(f"❌ Worker-{wid} hata: {e}", f"W{wid}", Fore.RED)
            continue

    # Bitiş sinyali
    try:
        out_q.put(None, timeout=1.0)
    except queue.Full:
        pass
        
    log(f"Worker-{wid} tamamlandı.", f"W{wid}", Fore.MAGENTA)

# -------------- OPTIMIZED WRITER --------------
def writer(in_q: Queue, stop_event: Event):
    try:
        store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)

        finished_workers = 0
        total_chunks_added = 0
        t0 = time.time()
        last_hb = t0
        last_flush = t0

        ACC_VEC, ACC_TXT, ACC_URL = [], [], []
        FLUSH_ADD_EVERY = 2048

        log("Writer başladı (optimized).", "WRITER", Fore.GREEN)

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
                        "metadata": {
                            "doc_type": DOC_TYPE,
                            "url": ACC_URL[j],
                        },
                    }
                    
            total_chunks_added += vecs.shape[0]
            ACC_VEC.clear()
            ACC_TXT.clear() 
            ACC_URL.clear()

        while finished_workers < N_WORKERS:
            current_time = time.time()
            
            # Periodic flush
            if current_time - last_flush > 30 and ACC_VEC:  # 30 saniyede bir flush
                flush_add()
                last_flush = current_time
                
            # Heartbeat
            if current_time - last_hb >= 30:
                ntotal = store.index.ntotal if store.index else 0
                dt = current_time - t0
                rps = total_chunks_added / max(dt, 1e-6)
                log(f"[HB] chunks={total_chunks_added} ntotal={ntotal} rate={rps:.1f} vec/s", 
                    "WRITER", Fore.CYAN)
                last_hb = current_time

            try:
                item = in_q.get(timeout=2.0)
            except queue.Empty:
                # Zaman aşımında flush kontrolü
                if ACC_VEC and sum(v.shape[0] for v in ACC_VEC) >= FLUSH_ADD_EVERY:
                    flush_add()
                continue

            if item == WRITER_POISON:
                break
                
            if item is None:
                finished_workers += 1
                log(f"Writer: {finished_workers}/{N_WORKERS} worker tamamlandı", "WRITER", Fore.GREEN)
                continue

            vecs, texts, urls = item
            ACC_VEC.append(np.asarray(vecs, dtype=np.float32))
            ACC_TXT.extend(texts)
            ACC_URL.extend(urls)

            # Batch flush
            if sum(v.shape[0] for v in ACC_VEC) >= FLUSH_ADD_EVERY:
                flush_add()

        # Final flush
        if ACC_VEC:
            flush_add()
            
        flush_atomic(store)

        dt = time.time() - t0
        rps = total_chunks_added / max(dt, 1e-6)
        ntotal = store.index.ntotal if store.index else 0
        
        log(f"✅ Writer tamamlandı. chunks={total_chunks_added} | {dt:.1f}s ~{rps:.1f} vec/s | ntotal={ntotal}",
            "WRITER", Fore.GREEN)

    except Exception as e:
        log(f"❌ Writer hatası: {e}", "WRITER", Fore.RED)
        import traceback
        traceback.print_exc()

# -------------- OPTIMIZED MAIN --------------
def main():
    try:
        set_start_method("fork", force=True)
    except RuntimeError:
        pass

    log(f"Sistem: {AVAILABLE_CPUS} CPU core | {N_WORKERS} worker", "SYS", Fore.CYAN)

    # Model preload
    preload_model()

    pf = ensure_processed_file_exists()
    log(f"Resume dosyası: {pf}", "SYS", Fore.CYAN)

    files = list_parquets(ROOT_DIR)
    if not files:
        log(f"❌ Parquet bulunamadı: {ROOT_DIR}", "SYS", Fore.RED)
        sys.exit(1)

    stop_event = Event()
    
    # Kuyruklar
    q_r2w  = Queue(maxsize=Q_R2W_SIZE)
    q_w2wr = Queue(maxsize=Q_W2WR_SIZE)

    # Processes
    p_reader = Process(target=reader, args=(files, q_r2w, stop_event), daemon=True)
    p_writer = Process(target=writer, args=(q_w2wr, stop_event), daemon=False)
    workers  = [Process(target=worker, args=(q_r2w, q_w2wr, stop_event, i+1), daemon=True)
                for i in range(N_WORKERS)]

    def graceful_shutdown(signum=None, frame=None):
        log(f"⚠️ Sinyal alındı ({signum}). Kapanış başlatılıyor…", "SYS", Fore.YELLOW)
        stop_event.set()
        
        # Kuyrukları temizle
        try:
            while not q_r2w.empty():
                q_r2w.get_nowait()
        except: pass
            
        try:
            q_w2wr.put_nowait(WRITER_POISON)
        except: pass

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(lambda: stop_event.set())

    t0 = time.time()
    
    # Başlatma sırası önemli
    p_writer.start()
    for p in workers: 
        p.start()
    p_reader.start()

    try:
        p_reader.join()
        log("Reader tamamlandı, worker'lar bekleniyor...", "SYS", Fore.CYAN)
        
        for p in workers: 
            p.join(timeout=30)  # Timeout ile
            
        log("Worker'lar tamamlandı, writer bekleniyor...", "SYS", Fore.CYAN)
        
        q_w2wr.put(WRITER_POISON, timeout=10)
        p_writer.join(timeout=30)

    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)
        
    finally:
        # Zorla temizlik
        for p in workers:
            if p.is_alive():
                p.terminate()
        if p_writer.is_alive():
            p_writer.terminate()
            
        log(f"🏁 Toplam süre: {time.time()-t0:.1f}s", "SYS", Fore.CYAN)

if __name__ == "__main__":
    main()