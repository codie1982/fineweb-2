from embedding_index import EmbeddingIndex
import numpy as np

# Aynı FAISS ve meta dosyalarını kullanıyoruz
INDEX_PATH = "faiss.index"
META_PATH = "meta.json"

# FAISS + model yükle
store = EmbeddingIndex(
    index_path=INDEX_PATH,
    meta_path=META_PATH,
)

# 1️⃣ Basit metinle arama
query = "Sağlık ve estetik uygulamaları"
print(f"\n🔎 Sorgu: {query}")
results = store.search(text=query, vector=None, k=5)

print(f"Toplam {len(results)} sonuç bulundu.\n")
for r in results:
    print(f"[score={r['score']:.4f}] {r['metadata'].get('url', '')}")
    snippet = (r['text'][:200] + "...") if len(r['text']) > 200 else r['text']
    print(f"  {snippet}\n")


# 2️⃣ Aynı sorguyu vektör üzerinden test et
print("\n⚙️ Vektör araması testi:")
vec = store.model.encode(query)
vec_results = store.search(text=None, vector=vec.tolist(), k=5)

for r in vec_results:
    print(f"[score={r['score']:.4f}] {r['metadata'].get('url', '')}")


# 3️⃣ Basit metadata filtresi (örnek)
print("\n🎯 'fineweb2' doc_type filtresiyle arama:")
filtered_results = store.search(
    text=query,
    vector=None,
    k=5,
    simple_filter={"metadata.doc_type": "fineweb2"}
)
for r in filtered_results:
    print(f"[score={r['score']:.4f}] {r['metadata'].get('url', '')}")
