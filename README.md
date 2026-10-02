# RAG Demo

一個使用 **Google Gemini** + **ChromaDB** 建立的 RAG（Retrieval-Augmented Generation）範例專案。

---

## 什麼是 RAG？

```
使用者提問
    ↓
向量資料庫檢索相關文件（Retrieval）
    ↓
問題 + 相關文件 → LLM 生成回答（Generation）
    ↓
回傳答案
```

---

## 專案結構

```
RAG/
├── knowledge_base/          # 📚 知識庫文件（放 .md 或 .pdf）
│   └── ai_knowledge.md      # 範例知識文件
├── chroma_db/               # 💾 向量資料庫（執行後自動建立）
│   ├── chroma.sqlite3       # 文字內容、Metadata
│   └── [uuid]/
│       └── data_level0.bin  # 實際向量數字（HNSW 索引）
├── 01_build_index.py        # 步驟 1：建立索引
├── 02_query.py              # 步驟 2：基礎問答
├── 03_advanced.py           # 步驟 3：進階多輪對話
├── inspect_db.py            # 查看 DB 內容工具
├── .env.example             # 環境變數範本
├── requirements.txt         # 套件清單
└── README.md
```

---

## 快速開始

### 1. 安裝套件

```bash
pip install -r requirements.txt

# 若需要支援 PDF：
pip install pypdf
```

### 2. 設定 API Key

```bash
# 複製範本
copy .env.example .env

# 編輯 .env，填入你的 Google API Key
# 前往 https://aistudio.google.com 免費申請
```

### 3. 建立索引

```bash
python 01_build_index.py
```

### 4. 開始問答

```bash
# 基礎版
python 02_query.py

# 進階版（多輪對話 + 相似度分數）
python 03_advanced.py
```

---

## 加入自己的知識

把你的 `.md`、`.txt` 或 `.pdf` 文件放入 `knowledge_base/` 資料夾，然後重新執行：

```bash
python 01_build_index.py
```

腳本會**自動清除舊 DB、重新建立**，把所有文件一起索引進去。

---

## 01_build_index.py 的概念

不只是「把文字寫入 DB」，它做了 4 件事：

```
文件 (PDF / MD)
  ↓ 1. 讀取
原始文字
  ↓ 2. 切片 (Chunking)  ← 每 500 字切一段，重疊 50 字保留上下文
["什麼是AI...", "什麼是RAG...", "台灣AI..."]
  ↓ 3. Embedding        ← 呼叫 Gemini API，把文字轉成數字向量
[[-0.022, 0.020, 0.017, ...],
 [-0.006, 0.010, 0.001, ...]]
  ↓ 4. 儲存
ChromaDB（文字 + 向量 都存進去）
```

### 為什麼要切片（Chunking）？

LLM 有 Token 上限，不能把整本書一次丟進去。切片後只取「最相關的幾段」送給 LLM，效率更高、回答更精準。

---

## 向量 DB vs 關聯式 DB

|  | 關聯式 DB（如 PostgreSQL） | 向量 DB（如 ChromaDB） |
|--|--|--|
| **存什麼** | 結構化欄位（姓名、金額、日期） | 文字 + 對應的數字向量 |
| **怎麼查詢** | `WHERE name = '王小明'`（精確符合） | 「找語意最相近的片段」 |
| **查詢邏輯** | 字串完全比對 | 向量空間距離計算 |
| **適合場景** | 訂單、帳號、交易記錄 | 文章搜尋、問答、推薦系統 |

---

## 向量的用意是什麼？

**把「語意」變成「距離」！**

同樣的意思，向量數字就會接近；不同主題，數字就差很遠：

```
"機器學習"  → [0.12, -0.34,  0.87, ...]
"AI 學習"   → [0.11, -0.33,  0.85, ...]  ← 數字很接近（語意相似）
"今天天氣"  → [0.89,  0.21, -0.44, ...]  ← 數字差很遠（語意不同）
```

當你問「什麼是 AI？」：
1. 問題也被轉成向量
2. 和 DB 裡每個片段的向量算距離
3. **距離最近 = 語意最相關** → 取出來給 LLM 生成答案

> 這就是為什麼你問「AI 是什麼」，它能找到寫著「人工智慧」的文章——  
> 因為**語意相同，向量就接近**，完全不需要關鍵字完全一樣！

---

## 測試問題範例

```
❓ 什麼是 RAG？
❓ 機器學習有哪些分類？
❓ 台灣有哪些 AI 相關企業？
❓ ChromaDB 和 FAISS 有什麼差別？
```

---

## 技術棧

| 元件 | 工具 |
|------|------|
| LLM | Google Gemini 3.8 Flash |
| Embedding | Google gemini-embedding-001 |
| 向量資料庫 | ChromaDB |
| 框架 | LangChain |

---

## 常見錯誤排除

| 錯誤訊息 | 原因 | 解法 |
|----------|------|------|
| `No module 'langchain.text_splitter'` | 新版拆成獨立套件 | 改用 `langchain_text_splitters` |
| `UnicodeEncodeError` emoji crash | Windows cp950 終端機 | 加 `sys.stdout.reconfigure(encoding='utf-8')` |
| `text-embedding-004` 404 | 模型已更名 | 改用 `models/gemini-embedding-001` |
| `gemini-2.0-flash` 404 | 模型已下線 | 改用 `gemini-3.8-flash` |

---

## 換成其他向量資料庫

LangChain 把所有向量 DB 包成**同樣的介面**，換 DB 只需要改 import 和初始化那幾行，`retriever`、`chain` 完全不用動。

```
你的程式碼
    ↓
LangChain 統一介面（.from_documents() / .as_retriever()）
    ↓
ChromaDB / FAISS / pgvector / Qdrant ... （底層互換）
```

### 各資料庫比較

| DB | 類型 | 安裝 | 適合場景 |
|----|------|------|----------|
| **ChromaDB** ← 目前使用 | 本地 | 已安裝 | Demo、學習 |
| **FAISS** | 本地函式庫 | `pip install faiss-cpu` | 研究、快速原型 |
| **pgvector** | PostgreSQL 擴充 | 需架 PostgreSQL | 生產環境、已有 PG |
| **Qdrant** | 本地 / 雲端 | `pip install qdrant-client` | 中大型專案、有 Web UI |
| **Pinecone** | 純雲端 | `pip install pinecone` | 雲端生產環境 |

### 換成 FAISS（最簡單，純本地）

```python
# pip install faiss-cpu

# 01_build_index.py
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
vectorstore.save_local("./faiss_db")

# 02_query.py
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local(
    "./faiss_db",
    embeddings,
    allow_dangerous_deserialization=True,
)
```

### 換成 pgvector（PostgreSQL，適合生產環境）

```python
# pip install psycopg2-binary pgvector langchain-postgres
# 需先在 PostgreSQL 執行：CREATE EXTENSION vector;

from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/mydb"

# 01_build_index.py
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_demo",
    connection=CONNECTION_STRING,
)

# 02_query.py
vectorstore = PGVector(
    collection_name="rag_demo",
    connection=CONNECTION_STRING,
    embeddings=embeddings,
)
```

### 換成 Qdrant（有 Web 管理介面）

```python
# pip install qdrant-client langchain-qdrant
# 先啟動 Qdrant：docker run -p 6333:6333 qdrant/qdrant

from langchain_qdrant import Qdrant

# 01_build_index.py
vectorstore = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="rag_demo",
)

# 02_query.py
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
vectorstore = Qdrant(client=client, collection_name="rag_demo", embeddings=embeddings)
```

### 建議選擇

| 情境 | 建議 |
|------|------|
| 繼續學習 / Demo | 維持 **ChromaDB**，夠用了 |
| 想要更快的搜尋速度 | 換 **FAISS** |
| 放到正式環境、有 PostgreSQL | 換 **pgvector** |
| 想要 Web UI 管理介面 | 換 **Qdrant** |

