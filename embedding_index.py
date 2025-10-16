# -*- coding: utf-8 -*-
"""
EmbeddingIndex – FAISS + SentenceTransformer tabanlı metin indeksleme sistemi
- Otomatik chunking (parametrik boyut & overlap)
- Her chunk için ayrı embedding
- Orijinal metni meta.json içinde saklama
- SHA1, URL, doc_type, chunk_index bilgileri dahil
"""

import os
import re
import json
import uuid
import time
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingIndex:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        index_path: str = "faiss.index",
        meta_path: str = "meta.json",
    ) -> None:
        self.model_name = model_name
        self.index_path = index_path
        self.meta_path = meta_path

        self._lock = threading.RLock()  # Reentrant lock for better performance
        self.index: Optional[faiss.IndexIDMap2] = None
        self.meta: Dict[int, Dict[str, Any]] = {}
        self._next_int_id: int = 1

        # Model loading optimized
        self.model = SentenceTransformer(self.model_name)
        self.model.eval()

        # FAISS thread settings
        try:
            faiss.omp_set_num_threads(min(4, os.cpu_count() // 2))
        except:
            pass

        try:
            self._load_state()
        except Exception as e:
            print(f"Initial load failed, starting fresh: {e}")

    def _ensure_documents_bucket(self) -> None:
        if "documents" not in self.meta or not isinstance(self.meta.get("documents"), dict):
            self.meta["documents"] = {}
    # =====================================================
    # ✅ GENEL API
    # =====================================================

    def upsert_vector(
            self,
            text: Optional[str],
            vector: Optional[List[float]] = None,
            external_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            int_id: Optional[int] = None,
            chunk_size: int = 1500,
            overlap: int = 200,
        ) -> Dict[str, Any]:
            if not text or not text.strip():
                raise ValueError("Provide 'text'.")

            full_text_sha1 = self._sha1_of(text)
            doc_id = f"doc_{full_text_sha1[:12]}"

            self._ensure_documents_bucket()
            if doc_id in self.meta["documents"]:
                return {"doc_ref": doc_id, "status": "exists"}

            # Chunking
            t_chunk0 = time.time()
            chunks = self._chunk_text_simple(text, size=chunk_size, overlap=overlap)
            t_chunk1 = time.time()
            if not chunks:
                raise ValueError("No chunks created from text.")

            # Store document
            self.meta["documents"][doc_id] = {
                "url": (metadata or {}).get("url"),
                "full_text_sha1": full_text_sha1,
                "full_text": text,
                "total_chunks": len(chunks),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Encode - batch optimization
            t_enc0 = time.time()
            chunk_texts = [c[0] for c in chunks]
            
            # Use larger batches for efficiency
            vecs = self.model.encode(
                chunk_texts, 
                batch_size=min(256, len(chunk_texts)),
                show_progress_bar=False,
                convert_to_tensor=False  # Direct numpy for better performance
            )
            vecs = np.array(vecs, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            t_enc1 = time.time()

            dim = vecs.shape[1]
            with self._lock:
                self._ensure_index(dim)
                self._normalize(vecs)

                # FAISS add - single operation
                t_add0 = time.time()
                start_id = self._next_int_id
                ids = np.arange(start_id, start_id + vecs.shape[0], dtype=np.int64)
                self._next_int_id += vecs.shape[0]
                
                self.index.add_with_ids(vecs, ids)

                # Batch metadata update
                for i, (txt, s, e, h_path) in enumerate(chunks):
                    faiss_id = int(ids[i])
                    self.meta[faiss_id] = {
                        "external_id": str(uuid.uuid4()),
                        "metadata": {
                            "doc_ref": doc_id,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "sha1": self._sha1_of(txt),
                            "h_path": h_path,
                            "char_start": s,
                            "char_end": e,
                        },
                    }
                t_add1 = time.time()

                self._save_state()

            return {
                "doc_ref": doc_id,
                "total_chunks": len(chunks),
                "faiss_ids": ids.tolist(),
                "timings": {
                    "chunk_ms": (t_chunk1 - t_chunk0) * 1000.0,
                    "encode_ms": (t_enc1 - t_enc0) * 1000.0,
                    "add_ms": (t_add1 - t_add0) * 1000.0,
                    "total_ms": (t_add1 - t_chunk0) * 1000.0,
                },
            }

    def search(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        k: int = 5,
        simple_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if vector is None:
            if not text or not text.strip():
                return []
            q = self.model.encode(text, show_progress_bar=False)
        else:
            q = np.array(vector, dtype=np.float32)

        if q.ndim == 1:
            q = q.reshape(1, -1)

        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            if self.index.d != q.shape[1]:
                raise ValueError(f"Query dim {q.shape[1]} != index dim {self.index.d}")

            self._normalize(q)
            scores, ids = self.index.search(q, int(k))

            results = []
            for i, idx in enumerate(ids[0]):
                if idx == -1:
                    continue
                meta = self.meta.get(int(idx))
                if not meta:
                    continue

                if simple_filter and not self._passes_filter(meta, simple_filter):
                    continue

                results.append({
                    "id": int(idx),
                    "score": float(scores[0][i]),
                    "external_id": meta.get("external_id"),
                    "text": meta.get("text"),
                    "metadata": meta.get("metadata", {}),
                })

        return results
    def ingest_markdown(self,url: str,raw_markdown: str,doc_type: str = "service",chunk_size: int = 1500,overlap: int = 200,) -> Dict[str, Any]:
        """
        Markdown dokümanı temizleyip upsert_vector ile indeksler.
        Full text sadece documents altında tutulur; chunk'lar doc_ref ile bağlanır.
        """
        if not self._is_valid_url(url):
            raise ValueError("invalid url")

        clean_md = self.clean_markdown(raw_markdown)
        if not clean_md.strip():
            raise ValueError("empty markdown after cleaning")

        # Başlık (opsiyonel)
        m = re.search(r"(?m)^#\s+(.+)$", clean_md)
        title = m.group(1).strip() if m else None

        # upsert_vector tüm chunk + meta işini halleder
        res = self.upsert_vector(
            text=clean_md,
            metadata={"url": url, "doc_type": doc_type.lower(), "title": title},
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Dönüş payload'ı (uyumlu ve özet)
        doc_ref = res.get("doc_ref")
        total_chunks = res.get("total_chunks")
        dim = self.index.d if self.index is not None else None

        return {
            "success": True,
            "page": {"url": url, "title": title, "language": "tr"},
            "doc_ref": doc_ref,
            "total_chunks": total_chunks,
            "model": {"name": self.model_name, "dim": dim, "emb_ver": time.strftime("%Y-%m-%d")},
        }

    # =====================================================
    # 🧠 TEXT YARDIMCILARI
    # =====================================================

    @staticmethod
    def clean_markdown(md: str) -> str:
        """Basit markdown temizleyici"""
        out = md.replace("\r\n", "\n").replace("\r", "\n")
        out = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", out)  # görselleri sil
        out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)  # linkleri text'e çevir
        out = re.sub(r"https?://\S+", "", out)  # çıplak URL sil
        return out.strip()


    def _chunk_text_simple(self, text: str, size: int = 1500, overlap: int = 200) -> List[Tuple[str, int, int, List[str]]]:
        """
        Düz metni sabit boyutlu parçalara böler.
        Her parça 4-tuple döner: (chunk_text, char_start, char_end, h_path)
        """
        chunks: List[Tuple[str, int, int, List[str]]] = []
        n = len(text)
        start = 0
        step = max(1, size - overlap)  # overlap >= size olursa kilitlenmesin

        while start < n:
            end = min(n, start + size)
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end, ["# Loose"]))
            start += step

        return chunks

    # =====================================================
    # 🔐 İÇ YARDIMCILAR
    # =====================================================

    @staticmethod
    def _normalize(a: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(a)
        return a

    def _save_state(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.meta.items()}, f, ensure_ascii=False, indent=2)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def _load_state(self) -> None:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.meta = {}
            for k, v in raw.items():
                try:
                    ik = int(k)
                    self.meta[ik] = v
                except Exception:
                    self.meta[k] = v
        else:
            self.meta = {}

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None

        # sadece sayısal id'lere bakarak next id belirle
        numeric_keys = [k for k in self.meta.keys() if isinstance(k, int)]
        self._next_int_id = (max(numeric_keys) + 1) if numeric_keys else 1


    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap2(base)
        elif self.index.d != dim:
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap2(base)
            self.meta.clear()
            self._next_int_id = 1
            self._save_state()

    @staticmethod
    def _sha1_of(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        return bool(url and re.match(r"^(?:http|https)://", url))

    @staticmethod
    def _passes_filter(item_meta: Dict[str, Any], filt: Dict[str, Any]) -> bool:
        for key, val in (filt or {}).items():
            if key.startswith("metadata."):
                sub = key.split(".", 1)[1]
                if (item_meta.get("metadata") or {}).get(sub) != val:
                    return False
            else:
                if item_meta.get(key) != val:
                    return False
        return True
