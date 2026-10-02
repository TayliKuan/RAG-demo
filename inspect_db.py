import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("rag_demo")

results = col.get(include=["documents", "metadatas", "embeddings"])

print("=== ChromaDB 完整資料結構 ===")
print(f"共 {len(results['ids'])} 筆\n")

for i in range(len(results["ids"])):
    print(f"--- 片段 {i+1} ---")
    print(f"ID      : {results['ids'][i]}")
    print(f"文字    : {results['documents'][i][:60]}...")
    print(f"Metadata: {results['metadatas'][i]}")
    emb = results["embeddings"][i]
    print(f"向量維度: {len(emb)} 維")
    print(f"前10個數字: {[round(v, 4) for v in emb[:10]]}")
    print(f"最大值={round(max(emb),4)}, 最小值={round(min(emb),4)}")
    print()

print("=== 實體檔案位置 ===")
for root, dirs, files in os.walk("./chroma_db"):
    level = root.replace("./chroma_db", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root) or 'chroma_db'}/")
    for f in files:
        size = os.path.getsize(os.path.join(root, f))
        print(f"{indent}  {f}  ({size:,} bytes)")
