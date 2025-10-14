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

        self._lock = threading.Lock()
        self.index: Optional[faiss.IndexIDMap2] = None
        self.meta: Dict[int, Dict[str, Any]] = {}
        self._next_int_id: int = 1

        self.model = SentenceTransformer(self.model_name)
        self.model.eval()

        try:
            self._load_state()
        except Exception:
            pass

    # =====================================================
    # ✅ GENEL API
    # =====================================================

    def upsert_vector(
        self,
        text: Optional[str],
        vector: Optional[List[float]] = None,
        external_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1500,
        overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Uzun bir text'i chunk'lara bölerek FAISS'e ekler.
        Her chunk ayrı embedding olarak kaydedilir.
        Orijinal metin metadata içinde 'full_text' olarak saklanır.
        """
        if not text or not text.strip():
            raise ValueError("Provide 'text' or 'vector'")

        full_text_sha1 = self._sha1_of(text)
        results = []

        # 1️⃣ Chunk'lara böl
        chunks = self._chunk_text_simple(text, size=chunk_size, overlap=overlap)
        total_chunks = len(chunks)

        with self._lock:
            for idx, chunk in enumerate(chunks):
                v = self.model.encode(chunk)
                if not isinstance(v, np.ndarray):
                    v = np.array(v, dtype=np.float32)
                if v.ndim == 1:
                    v = v.reshape(1, -1)

                dim = v.shape[1]
                self._ensure_index(dim)
                self._normalize(v)

                chunk_id = self._next_int_id
                self._next_int_id += 1
                self.index.add_with_ids(v, np.array([chunk_id], dtype=np.int64))

                # 🔎 meta bilgisi
                self.meta[chunk_id] = {
                    "external_id": external_id or str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "full_text_sha1": full_text_sha1,
                        "full_text": text,
                        "sha1": self._sha1_of(chunk),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }

                results.append({
                    "id": chunk_id,
                    "external_id": self.meta[chunk_id]["external_id"],
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                })

            self._save_state()

        return results

    def search(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        k: int = 5,
        simple_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """KNN benzerlik araması yapar."""
        if vector is None:
            if not text or not text.strip():
                return []
            q = self.model.encode(text)
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

    def ingest_markdown(
        self,
        url: str,
        raw_markdown: str,
        doc_type: str = "service",
        chunk_size: int = 1500,
        overlap: int = 200,
    ) -> Dict[str, Any]:
        """
        Markdown dokümanı temizleyip chunk'lara böler ve FAISS'e ekler.
        """
        if not self._is_valid_url(url):
            raise ValueError("invalid url")

        clean_md = self.clean_markdown(raw_markdown)
        content_hash = self._sha1_of(clean_md)
        title = None
        m = re.search(r"(?m)^#\s+(.+)$", clean_md)
        if m:
            title = m.group(1).strip()

        # 🔥 Chunk işlemi burada
        chunks = self._chunk_text_simple(clean_md, size=chunk_size, overlap=overlap)
        total_chunks = len(chunks)

        results = []
        with self._lock:
            for idx, chunk in enumerate(chunks):
                v = self.model.encode(chunk)
                if v.ndim == 1:
                    v = v.reshape(1, -1)

                dim = v.shape[1]
                self._ensure_index(dim)
                self._normalize(v)

                cid = self._next_int_id
                self._next_int_id += 1
                self.index.add_with_ids(v, np.array([cid], dtype=np.int64))

                self.meta[cid] = {
                    "external_id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        "doc_type": doc_type.lower(),
                        "url": url,
                        "title": title,
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "full_text": clean_md,
                        "full_text_sha1": content_hash,
                        "sha1": self._sha1_of(chunk),
                    },
                }

                results.append({
                    "chunk_index": idx,
                    "faiss_id": cid,
                    "url": url,
                    "text": chunk,
                })

            self._save_state()

        return {
            "success": True,
            "page": {"url": url, "title": title, "language": "tr", "content_hash": content_hash},
            "chunks": results,
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

    @staticmethod
    def _chunk_text_simple(text: str, size: int = 1500, overlap: int = 200) -> List[str]:
        """Uzun text'i sabit uzunlukta parçalara böler."""
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            chunks.append(text[start:end])
            start += size - overlap
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
                self.meta = {int(k): v for k, v in json.load(f).items()}
        else:
            self.meta = {}

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None

        self._next_int_id = (max(self.meta.keys()) + 1) if self.meta else 1

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
