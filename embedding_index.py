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
    """
    Tüm embedding + FAISS + metin işleme mantığını kapsar.
    Flask router'ları bu sınıfı kullanır.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        index_path: str = "faiss.index",
        meta_path: str = "meta.json",
    ) -> None:
        self.model_name = model_name
        self.index_path = index_path
        self.meta_path = meta_path

        # Paylaşımlı kilit; Flask çoklu thread altında güvenli erişim
        self._lock = threading.Lock()

        # FAISS + meta durumları
        self.index: Optional[faiss.IndexIDMap2] = None
        self.meta: Dict[int, Dict[str, Any]] = {}
        self._next_int_id: int = 1

        # Modeli baştan yükle
        self.model = SentenceTransformer(self.model_name)
        self.model.eval()
        # Kalıcı durumu getir
        try:
            self._load_state()
        except Exception:
            # İlk kurulumda dosyalar yoksa sessizce devam
            pass

    # ---------- Public API ----------

    def vectorize_text(self, text: str) -> List[float]:
        """Metni embed edip Python list olarak döndürür."""
        if not text or not text.strip():
            raise ValueError("No text provided")
        vec = self.model.encode(text)
        return vec.tolist()

    def upsert_vector(
        self,
        text: Optional[str],
        vector: Optional[List[float]],
        external_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        int_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tek bir vektörü FAISS'e ekler/günceller.
        text verilirse encode edilir; vector verilirse direkt kullanılır.
        """
        if vector is None:
            if not text or not text.strip():
                raise ValueError("Provide either 'text' or 'vector'.")
            v = self.model.encode(text)
        else:
            v = np.array(vector, dtype=np.float32)

        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1).astype(np.float32)

        dim = v.shape[1]

        with self._lock:
            self._ensure_index(dim)
            self._normalize(v)

            if int_id is None:
                int_id = self._next_int_id
                self._next_int_id += 1
            else:
                # Varsa eski kaydı çıkar (yoksa FAISS zaten ignore eder)
                try:
                    self.index.remove_ids(np.array([int(int_id)], dtype=np.int64))
                except Exception:
                    pass

            self.index.add_with_ids(v, np.array([int(int_id)], dtype=np.int64))
            self.meta[int(int_id)] = {
                "external_id": external_id or str(uuid.uuid4()),
                "text": text if text else self.meta.get(int(int_id), {}).get("text"),
                "metadata": metadata or {},
            }
            self._save_state()

        return {"id": int(int_id), "external_id": self.meta[int(int_id)]["external_id"]}

    def search(
        self,
        text: Optional[str],
        vector: Optional[List[float]],
        k: int,
        simple_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Metin veya vektörle benzerlik araması yapar, basit metadata filtresi uygular."""
        if vector is None:
            if not text or not text.strip():
                return []
            q = self.model.encode(text)
        else:
            q = np.array(vector, dtype=np.float32)

        if not isinstance(q, np.ndarray):
            q = np.array(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1).astype(np.float32)

        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            if self.index.d != q.shape[1]:
                raise ValueError(f"Query dim {q.shape[1]} != index dim {self.index.d}")

            self._normalize(q)
            scores, ids = self.index.search(q, int(k))
            ids = ids[0].tolist()
            scores = scores[0].tolist()

            out = []
            for i, idx in enumerate(ids):
                if idx == -1:
                    continue
                m = self.meta.get(int(idx))
                if not m:
                    continue

                if simple_filter:
                    if not self._passes_filter(m, simple_filter):
                        continue

                out.append(
                    {
                        "id": int(idx),
                        "score": float(scores[i]),
                        "external_id": m.get("external_id"),
                        "text": m.get("text"),
                        "metadata": m.get("metadata", {}),
                    }
                )
        return out

    def ingest_markdown(
        self,
        url: str,
        raw_markdown: str,
        doc_type: str = "service",
        target_chars: int = 1100,
        overlap_chars: int = 180,
    ) -> Dict[str, Any]:
        """
        Markdown -> clean -> chunk -> embed -> FAISS add_with_ids
        """
        if not self._is_valid_url(url):
            raise ValueError("invalid url")
        if not raw_markdown or not raw_markdown.strip():
            raise ValueError("empty markdown")

        clean_md = self.clean_markdown(raw_markdown)
        content_hash = self._sha1_of(clean_md)
        title = None
        m = re.search(r"(?m)^#\s+(.+)$", clean_md)
        if m:
            title = m.group(1).strip()

        raw_chunks = self.chunk_text(clean_md, target=target_chars, overlap=overlap_chars)

        results = []
        with self._lock:
            if not raw_chunks:
                page = {"url": url, "title": title, "language": "tr", "content_hash": content_hash}
                return {
                    "success": True,
                    "page": page,
                    "chunks": [],
                    "model": {"name": self.model_name},
                    "index": {"name": "main_flat_ip"},
                }

            texts = [c[0] for c in raw_chunks]
            vecs = self.model.encode(texts)
            vecs = np.array(vecs, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)

            dim = vecs.shape[1]
            self._ensure_index(dim)
            self._normalize(vecs)

            start_id = self._next_int_id
            ids = np.arange(start_id, start_id + vecs.shape[0], dtype=np.int64)
            self._next_int_id += vecs.shape[0]
            self.index.add_with_ids(vecs, ids)

            for i, (txt, s, e, h_path) in enumerate(raw_chunks):
                faiss_id = int(ids[i])
                chunk_id = str(uuid.uuid4())
                self.meta[faiss_id] = {
                    "external_id": chunk_id,
                    "text": txt,
                    "metadata": {"doc_type": doc_type.lower(), "url": url, "h_path": h_path},
                }
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "faiss_id": faiss_id,
                        "url": url,
                        "h_path": h_path,
                        "text": txt,
                        "char_start": s,
                        "char_end": e,
                        "metadata": {"doc_type": doc_type.lower()},
                    }
                )

            self._save_state()

        page = {"url": url, "title": title, "language": "tr", "content_hash": content_hash}
        return {
            "success": True,
            "page": page,
            "chunks": results,
            "model": {"name": self.model_name, "dim": int(vecs.shape[1]), "emb_ver": time.strftime("%Y-%m-%d")},
            "index": {"name": "main_flat_ip"},
        }

    # ---------- Text utils (public) ----------

    @staticmethod
    def clean_markdown(md: str, one_per_line: bool = True, drop_footers: bool = True) -> str:
        """Mevcut temizleyicinin birebir taşınmış versiyonu."""
        if not isinstance(md, str):
            return ""

        out = md.replace("\r\n", "\n").replace("\r", "\n").strip()

        # (<https://…>) → (https://…)
        out = re.sub(r"\(<\s*([^>]+)\s*>\)", r"(\1)", out, flags=re.I)
        # https:/ → https://
        out = re.sub(r"\b(https?):\/(?!\/)", r"\1://", out, flags=re.I)
        # [12] başlıkları
        out = re.sub(r"^\s*\[\d+\]\s*", "", out, flags=re.M)

        # javascript:void(0) ve hash-only
        def _rm_js_void(m):
            txt = m.group(1) or ""
            return txt.strip() if txt.strip() else ""

        out = re.sub(r"\[([^\]]*)\]\(\s*javascript:void\(0\)\s*\)", _rm_js_void, out, flags=re.I)

        def _rm_hash_or_empty(m):
            txt = m.group(1) or ""
            return txt.strip() if txt.strip() else ""

        out = re.sub(r"\[([^\]]+)\]\(\s*(#|\s*)\)", _rm_hash_or_empty, out)

        # [](...)
        out = re.sub(r"\[\s*\]\(\s*[^)]+\s*\)", "", out)

        # [text](url) → "text" (opsiyonel satır başı)
        def _link_to_text(m):
            txt = m.group(1)
            return f"\n{txt}\n" if one_per_line else txt

        out = re.sub(r"\[([^\]]+)\]\(\s*[^)]+\s*\)", _link_to_text, out)

        # resimleri kaldır
        out = re.sub(r"!\[[^\]]*\]\(\s*[^)]+\s*\)", "", out)
        # çıplak URL'leri sil
        out = re.sub(r"\bhttps?://\S+", "", out, flags=re.I)

        # newline/boşluk düzeltmeleri
        out = re.sub(r"[ \t]+\n", "\n", out)
        out = re.sub(r"\n[ \t]+", "\n", out)

        lines = [l.strip() for l in out.split("\n")]

        if drop_footers:
            footer_or_noise = re.compile(
                "|".join(
                    [
                        r"^gizlilik politikası$",
                        r"^kvkk$",
                        r"^kullanım koşulları$",
                        r"^çerez politikası$",
                        r"^privacy policy$",
                        r"^terms(?: and conditions)?$",
                        r"^cookies?$",
                        r"^iletişim(?: bilgilerimiz)?$",
                        r"^contact$",
                        r"^(tel:|telefon:|phone:|mailto:|e-?posta:)",
                        r"(mah\.|mahalle|sokak|cad\.|no:|pk:|kat\b|daire\b|istanbul|ankara|izmir)",
                        r"^copyright|^©",
                    ]
                ),
                flags=re.I,
            )
            lines = [l for l in lines if l and not footer_or_noise.search(l) and len(l) > 2]
        else:
            lines = [l for l in lines if l]

        # case-insensitive dedup
        seen = set()
        deduped = []
        for l in lines:
            key = l.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(l)

        result = "\n".join(deduped)
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        return result

    @staticmethod
    def chunk_text(text: str, target: int = 1100, overlap: int = 180) -> List[Tuple[str, int, int, List[str]]]:
        """Başlık+paragraf odaklı chunking."""
        parts = re.split(r"(?m)^(#{1,6}\s.+)$", text)
        blocks = []
        for i in range(0, len(parts), 2):
            pre = parts[i]
            hdr = parts[i + 1] if i + 1 < len(parts) else None
            body = parts[i + 2] if i + 2 < len(parts) else ""
            if hdr:
                blocks.append((hdr.strip(), body))
            elif pre.strip():
                blocks.append(("# Loose", pre))

        chunks = []
        cursor = 0
        for hdr, body in blocks:
            h_path = [hdr.lstrip("# ").strip()]
            paras = [p.strip() for p in body.split("\n\n") if p.strip()]
            buf = hdr + "\n"
            start = cursor
            for p in paras:
                if len(buf) + len(p) + 2 <= target:
                    buf += "\n" + p
                else:
                    end = start + len(buf)
                    chunks.append((buf.strip(), start, end, h_path))
                    tail = buf[-overlap:] if overlap > 0 else ""
                    buf = (hdr + "\n" + tail + p)[-target:]
                    start = end - len(buf)
            if buf.strip():
                end = start + len(buf)
                chunks.append((buf.strip(), start, end, h_path))
                cursor = end
        return chunks

    # ---------- Internal helpers ----------

    @staticmethod
    def _normalize(a: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(a)
        return a

    def _save_state(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.meta.items()}, f, ensure_ascii=False)
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
            base = faiss.IndexFlatIP(dim)  # cosine için IP + normalize
            self.index = faiss.IndexIDMap2(base)
        elif self.index.d != dim:
            # Model boyutu değişmiş → temiz kurulum
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
        url_regex = re.compile(r"^(?:http|https)://(?:\S+)")
        return bool(url and re.match(url_regex, url))

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