# -*- coding: utf-8 -*-
"""
Optimized multiprocessing ingest - CPU verimliliği artırıldı
- Reader: paketli dispatch
- Worker: chunk + büyük batch encode (convert_to_numpy=True)
- Writer: toplu FAISS add, atomik save, dosya boyutlu heartbeat
"""

import os, glob, queue, time, datetime, sys, signal, atexit, json, hashlib
from typing import List, Optional, Set, Tuple
from multiprocessing import Process, Queue, Event, set_start_method, cpu_count

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

AVAILABLE_CPUS   = cpu_count()
N_WORKERS        = min(max(AVAILABLE_CPUS - 2, 2), 16)  # 2 çekirdek reader+writer
if N_WORKERS < 4:
    N_WORKERS = 4

LIMIT_TOTAL      = 100                  # Tüm dataset için None yap
BATCH_READ       = 8192
MIN_CHARS        = 80
MAX_CHARS        = 8000
DOC_TYPE         = "fineweb2"
TEXT_CANDS       = ["text","content","document","page_content","raw_content","body","clean_text","html_text","markdown"]
URL_CANDS        = ["url","source_url","link","origin","canonical_url"]

# Kuyruk boyutları
Q_R2W_SIZE       = N_WORKERS * 500
Q_W2WR_SIZE      = N_WORKERS * 300

LOG_FILE         = None
WRITER_POISON    = "__WRITER_POISON__"

# Throughput
READ_DISPATCH    = 8192
CHUNK_SIZE       = 1500
CHUNK_OVERLAP    = 200
ENC_BATCH        = 256                # CPU için 128–512 arası deneyebilirsiniz

# Zamanlamalar
MAX_QUEUE_WAIT   = 0.2                # saniye
HEARTBEAT_S      = 30
PERIODIC_FLUSH_S = 30
# =====================================

# Global (fork öncesi preload için)
MODEL: Optional[SentenceTransformer] = None

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
    return f"{n:.1f}ZB"

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
    """meta.json ve faiss.index’i temp’e yazıp atomik replace."""
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
    """Fork öncesi modeli yükle (Linux’ta bellek paylaşımı)."""
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        torch.set_num_threads(1)  # her worker kendi thread’ini ayarlayacak
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = SentenceTransformer(MODEL_NAME, device=device)
    MODEL.eval()
    log(f"Encoder hazır: {MODEL_NAME} | device={device} | workers={N_WORKERS}", "SYS", Fore.CYAN)
    return MODEL

# ---------------- READER ----------------
def reader(files: List[str], out_q: Queue, stop_event: Event):
    sent = 0
    total_limit = LIMIT_TOTAL if isinstance(LIMIT_TOTAL, int) else float("inf")
    seen_files = load_processed()
    todo = [p for p in files if p not in seen_files]
    log(f"Reader başladı: {len(files)} dosya (atlanan={len(seen_files)}, işlenecek={len(todo)}).", "READER", Fore.BLUE)

    batch_T: List[str] = []
    batch_U: List[Optional[str]] = []
    batch_count = 0

    def flush_reader_batch(force=False) -> bool:
        nonlocal batch_T, batch_U, sent, batch_count
        if batch_count == 0:
            return True
        if not force and batch_count < READ_DISPATCH:
            return True
        try:
            out_q.put((batch_T.copy(), batch_U.copy()), timeout=MAX_QUEUE_WAIT)
            sent += batch_count
            log(f"Reader: {batch_count} satır gönderildi (toplam: {sent})", "READER", Fore.BLUE)
            batch_T.clear(); batch_U.clear(); batch_count = 0
            return True
        except queue.Full:
            log("⚠️ Reader queue dolu, bekleniyor...", "READER", Fore.YELLOW)
            return False

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

        for b in scanner.to_batches():
            if stop_event.is_set() or sent >= total_limit: break
            d = b.to_pydict()
            texts = d.get(tcol, [])
            urls  = d.get(ucol, []) if ucol else [None]*len(texts)

            for i, t in enumerate(texts):
                if stop_event.is_set() or sent >= total_limit: break
                if not t: continue
                s = str(t).strip()
                if not (MIN_CHARS <= len(s) <= MAX_CHARS): continue
                u = urls[i] if urls and urls[i] else f"http://fw2.local/{os.path.basename(path)}-{i}"

                batch_T.append(s)
                batch_U.append(u)
                batch_count += 1

                if batch_count >= READ_DISPATCH:
                    while not flush_reader_batch(force=True):
                        time.sleep(0.05)

        # dosya bitişinde elde kalanları gönder
        flush_reader_batch(force=True)
        mark_processed(path)

    # son flush
    flush_reader_batch(force=True)

    # worker’lara bitiş
    for _ in range(N_WORKERS):
        try:
            out_q.put(None, timeout=1.0)
        except queue.Full:
            pass

    log(f"✅ Reader bitti. Toplam {sent} satır gönderildi.", "READER", Fore.BLUE)

# ---------------- WORKER ----------------
def _chunk_simple(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    n = len(s)
    step = max(1, size - overlap)
    for start in range(0, n, step):
        end = min(n, start + size)
        yield s[start:end]

def worker(in_q, out_q, stop_event, wid: int):
    import gc
    # her worker tek thread
    try:
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"

    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME, device=device)
        MODEL.eval()
    model = MODEL

    log(f"Worker-{wid} başladı. device={device}", f"W{wid}", Fore.MAGENTA)

    processed_docs = 0
    t_last = time.time()

    while not stop_event.is_set():
        try:
            item = in_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:
            break

        texts, urls = item
        processed_docs += len(texts)

        # paket içindeki metinleri chunk’la
        batch_texts: List[str] = []
        batch_urls:  List[Optional[str]] = []
        for s, u in zip(texts, urls):
            if not s: continue
            s = s.strip()
            if not s: continue
            for ch in _chunk_simple(s):
                if len(ch) >= MIN_CHARS:
                    batch_texts.append(ch)
                    batch_urls.append(u)

        if not batch_texts:
            continue

        # encode -> garantili numpy float32
        vecs_out: List[np.ndarray] = []
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(batch_texts), ENC_BATCH):
                sub = batch_texts[i:i+ENC_BATCH]
                v = model.encode(
                    sub,
                    batch_size=ENC_BATCH,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True   # <-- kritik
                )
                vecs_out.append(np.asarray(v, dtype=np.float32))
        vecs = np.vstack(vecs_out) if len(vecs_out) > 1 else np.asarray(vecs_out[0], dtype=np.float32)
        dt = time.time() - t0

        # writer’a gönder
        try:
            out_q.put((vecs, batch_texts, batch_urls), timeout=MAX_QUEUE_WAIT)
        except queue.Full:
            log(f"⚠️ Worker-{wid}: writer queue dolu, tekrar denenecek", f"W{wid}", Fore.YELLOW)
            time.sleep(0.1)
            try:
                out_q.put((vecs, batch_texts, batch_urls), timeout=MAX_QUEUE_WAIT)
            except queue.Full:
                log(f"❌ Worker-{wid}: writer queue sürekli dolu, batch düşürüldü", f"W{wid}", Fore.RED)

        # performans logu (30 sn)
        now = time.time()
        if now - t_last >= 30:
            log(f"Worker-{wid}: batch_chunks={len(batch_texts)} enc={dt*1000:.0f}ms "
                f"(~{len(batch_texts)/max(dt,1e-6):.1f} vec/s) | docs_in_packet={len(texts)}",
                f"W{wid}", Fore.MAGENTA)
            t_last = now

        del vecs_out
        gc.collect()

    # bitiş sinyali
    try:
        out_q.put(None, timeout=1.0)
    except queue.Full:
        pass
    log(f"Worker-{wid} tamamlandı.", f"W{wid}", Fore.MAGENTA)

# ---------------- WRITER ----------------
def writer(in_q: Queue, stop_event: Event):
    try:
        # FAISS/OpenMP threadlerini sınırlayalım
        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        store = EmbeddingIndex(model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH)
        initial_count = store.index.ntotal if store.index else 0
        log(f"Writer başladı. CWD={os.getcwd()} | INDEX={INDEX_PATH} | META={META_PATH} | ntotal={initial_count}",
            "WRITER", Fore.GREEN)

        finished = 0
        total_chunks_added = 0
        t0 = time.time()
        last_hb = t0
        last_periodic = t0

        ACC_VEC: List[np.ndarray] = []
        ACC_TXT: List[str] = []
        ACC_URL: List[Optional[str]] = []
        FLUSH_ADD_EVERY = 2048

        def flush_add():
            nonlocal ACC_VEC, ACC_TXT, ACC_URL, total_chunks_added
            if not ACC_VEC:
                return
            try:
                vecs = np.vstack(ACC_VEC).astype(np.float32, copy=False)
            except ValueError as e:
                log(f"❌ Writer vstack hata: {e} | shapes={[v.shape for v in ACC_VEC]}", "WRITER", Fore.RED)
                ACC_VEC.clear(); ACC_TXT.clear(); ACC_URL.clear()
                return

            dim = vecs.shape[1]
            n = vecs.shape[0]

            with store._lock:
                store._ensure_index(dim)
                store._normalize(vecs)
                start_id = store._next_int_id
                ids = np.arange(start_id, start_id + n, dtype=np.int64)
                store._next_int_id += n
                store.index.add_with_ids(vecs, ids)

                # hafif meta (text istersen tut; yer kazanmak için kapatılabilir)
                for j, fid in enumerate(ids):
                    store.meta[int(fid)] = {
                        "external_id": os.urandom(8).hex(),
                        # "text": ACC_TXT[j],  # yerden kazanmak için yorumda bırak
                        "metadata": {
                            "doc_type": DOC_TYPE,
                            "url": ACC_URL[j],
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    }

                # hemen kaydet (kesinti riskine karşı)
                store._save_state()

            total_chunks_added += n
            log(f"Writer: flush_add -> {n} chunk eklendi (toplam: {total_chunks_added})",
                "WRITER", Fore.GREEN)
            ACC_VEC.clear(); ACC_TXT.clear(); ACC_URL.clear()

        while finished < N_WORKERS and not stop_event.is_set():
            now = time.time()

            if now - last_periodic >= PERIODIC_FLUSH_S and ACC_VEC:
                flush_add()
                last_periodic = now

            if now - last_hb >= HEARTBEAT_S:
                ntotal = store.index.ntotal if store.index else 0
                meta_size = os.path.getsize(META_PATH) if os.path.exists(META_PATH) else 0
                idx_size  = os.path.getsize(INDEX_PATH) if os.path.exists(INDEX_PATH) else 0
                dt = now - t0
                rps = total_chunks_added / max(dt, 1e-6)
                log(f"[HB] chunks={total_chunks_added} ntotal={ntotal} rate={rps:.1f} vec/s "
                    f"meta={_fmt_bytes(meta_size)} index={_fmt_bytes(idx_size)}",
                    "WRITER", Fore.CYAN)
                last_hb = now

            try:
                item = in_q.get(timeout=2.0)
            except queue.Empty:
                # pencere dolduysa flush
                if ACC_VEC and sum(v.shape[0] for v in ACC_VEC) >= FLUSH_ADD_EVERY:
                    flush_add()
                continue

            if item == WRITER_POISON:
                log("Writer: poison alındı", "WRITER", Fore.YELLOW)
                break

            if item is None:
                finished += 1
                log(f"Writer: {finished}/{N_WORKERS} worker tamamlandı", "WRITER", Fore.GREEN)
                continue

            vecs, texts, urls = item

            # koruyucu kontroller
            if not isinstance(vecs, np.ndarray) or vecs.size == 0:
                log("⚠️ Writer: boş/uygunsuz vecs geldi, atlandı", "WRITER", Fore.YELLOW)
                continue
            if len(texts) != vecs.shape[0] or len(urls) != vecs.shape[0]:
                log(f"⚠️ Writer: len mismatch vecs={vecs.shape[0]} texts={len(texts)} urls={len(urls)}, kısaltılıyor",
                    "WRITER", Fore.YELLOW)
                texts = texts[:vecs.shape[0]]
                urls  = urls[:vecs.shape[0]]

            ACC_VEC.append(np.asarray(vecs, dtype=np.float32))
            ACC_TXT.extend(texts)
            ACC_URL.extend(urls)

            acc_n = sum(v.shape[0] for v in ACC_VEC)
            log(f"Writer: batch alındı -> {vecs.shape[0]} vektör, acc={acc_n}", "WRITER", Fore.CYAN)

            if acc_n >= FLUSH_ADD_EVERY:
                flush_add()

        # kalanlar
        if ACC_VEC:
            flush_add()

        # son atomik yazım
        try:
            with store._lock:
                flush_atomic(store)
        except Exception as e:
            log(f"⚠️ Final flush_atomic hata: {e}", "WRITER", Fore.YELLOW)

        dt = time.time() - t0
        rps = total_chunks_added / max(dt, 1e-6)
        ntotal = store.index.ntotal if store.index else 0
        log(f"✅ Writer tamamlandı. chunks={total_chunks_added} | {dt:.1f}s ~{rps:.1f} vec/s | ntotal={ntotal}",
            "WRITER", Fore.GREEN)

    except Exception as e:
        log(f"❌ Writer hatası: {e}", "WRITER", Fore.RED)
        import traceback; traceback.print_exc()

# ---------------- MAIN ----------------
def main():
    try:
        set_start_method("fork", force=True)
    except RuntimeError:
        pass

    log(f"Sistem: {AVAILABLE_CPUS} CPU core | {N_WORKERS} worker", "SYS", Fore.CYAN)

    # model preload (Linux/Unix’te bellek paylaşımı)
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
        log(f"⚠️ Sinyal alındı ({signum}). Kapanış…", "SYS", Fore.YELLOW)
        stop_event.set()
        try:
            while not q_r2w.empty():
                q_r2w.get_nowait()
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

    # başlatma sırası
    p_writer.start()
    for p in workers: p.start()
    p_reader.start()

    try:
        p_reader.join()
        log("Reader tamamlandı, worker'lar bekleniyor…", "SYS", Fore.CYAN)

        for p in workers:
            p.join()

        log("Worker'lar tamamlandı, writer bekleniyor…", "SYS", Fore.CYAN)
        try:
            q_w2wr.put(WRITER_POISON, timeout=5)
        except Exception:
            pass
        p_writer.join()

    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)

    finally:
        # zorla temizlik
        for p in workers:
            if p.is_alive():
                p.terminate()
        if p_writer.is_alive():
            p_writer.terminate()

        dt = time.time() - t0
        log(f"🏁 Toplam süre: {dt:.1f}s", "SYS", Fore.CYAN)

if __name__ == "__main__":
    main()