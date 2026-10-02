# RAG Demo 🚀

一個使用 **Google Gemini** + **ChromaDB** 建立的 RAG（Retrieval-Augmented Generation）與 **Agent 智慧搜尋** 範例專案。

---

## 📌 什麼是 RAG 與 Agent？

### 1. 傳統 RAG 流程（檢索增強生成）
適合查詢**專屬私有知識庫**（如內部文件、規章、專案紀錄），避免 LLM 產生幻覺：
```
使用者提問
    ↓
向量資料庫比對檢索（Retrieval，找出最相關片段）
    ↓
問題 + 參考文件片段 → Gemini LLM 生成回答（Generation）
    ↓
輸出準確答案（附帶來源）
```

### 2. Agent + 網路搜尋流程（自動決策）
結合**本地知識庫**與**即時網路搜尋**，由 AI 自行判斷要查哪裡：
```
使用者提問
    ↓
Agent (ReAct 思考迴圈) 判斷資訊來源
    ├─► 知識庫內容 (AI、機器學習、內部文件等) ──► 查詢 ChromaDB
    └─► 即時/外部內容 (新聞、天氣、運動規則、即時事件) ──► 呼叫 DuckDuckGo 搜尋
    ↓
整合所有資料，生成最終回答
```

---

## 📂 專案結構

```
RAG/
├── knowledge_base/          # 📚 知識庫文件目錄（支援 .md、.txt、.pdf）
│   └── ai_knowledge.md      # 範例知識文件
├── chroma_db/               # 💾 向量資料庫（執行 01 後自動建立，請勿 commit）
│   ├── chroma.sqlite3       # 文字內容、Metadata
│   └── [uuid]/
│       └── data_level0.bin  # 實際向量數字（HNSW 索引結構）
├── 01_build_index.py        # 步驟 1：建立向量索引（支援 MD 與 PDF）
├── 02_query.py              # 步驟 2：基礎 RAG 問答
├── 03_advanced.py           # 步驟 3：進階問答（多輪對話記憶 + 相似度評分）
├── 04_agent_search.py       # 步驟 4：Agent 智慧搜尋（知識庫 + DuckDuckGo 即時聯網）
├── inspect_db.py            # 工具：查看 ChromaDB 底層向量與資料內容
├── .env.example             # 環境變數範本（供他人複製使用）
├── .gitignore               # Git 忽略設定（排除 .env 與 DB）
├── requirements.txt         # 核心相依套件清單
└── README.md                # 專案說明與常見問題解答
```

---

## ⚡ 快速開始

### 1. 安裝環境套件

```bash
pip install -r requirements.txt

# 支援 PDF 讀取與網路搜尋：
pip install pypdf ddgs duckduckgo-search
```

### 2. 設定 Google API Key

1. 前往 [Google AI Studio](https://aistudio.google.com) 免費申請 API Key。
2. 複製設定檔範本並填入金鑰：

```bash
# Windows 命令提示字元 (CMD)
copy .env.example .env
```

編輯 `.env` 檔案：
```env
GOOGLE_API_KEY=你的_GEMINI_API_KEY
```

### 3. 建立向量索引庫

```bash
python 01_build_index.py
```
> 執行後會自動解析 `knowledge_base/` 內的所有文件，切片、計算向量並儲存至 `./chroma_db`。

### 4. 執行問答程式

依照需求選擇不同的問答模式：

| 模式 | 指令 | 說明 |
|------|------|------|
| **基礎問答** | `python 02_query.py` | 單問單答，精準鎖定知識庫內容 |
| **進階對話** | `python 03_advanced.py` | 具備多輪上下文記憶、顯示各片段相似度分數 |
| **智慧 Agent** | `python 04_agent_search.py` | **知識庫 + 聯網搜尋**，可回答未收錄的即時問題 |

---

## 💡 核心技術概念與常見問題 (FAQ)

### Q1. `01_build_index.py` 是什麼原理？只是把文字存入資料庫嗎？
不只是存字，而是執行了 4 個關鍵步驟：
1. **讀取 (Load)**：支援 `.md` 與 `.pdf`，將文字分頁讀入。
2. **切片 (Chunking)**：以 500 字為單位切片（重疊 50 字保持語意完整）。因為 LLM 有單次 Context 限制，切片後只提取最關鍵的段落，效率與精確度最高。
3. **向量化 (Embedding)**：呼叫 Google Embedding 模型（如 `gemini-embedding-001`），將每一段文字轉成長度 3072 維的數字向量。
4. **建立索引與儲存 (Indexing & Store)**：寫入 ChromaDB，利用 HNSW 演算法建立鄰近圖索引以利極速搜尋。

---

### Q2. 向量資料庫與一般關聯式資料庫（如 MySQL / PostgreSQL）差在哪？為什麼需要向量？
- **關聯式 DB**：適合「精確比對」（例如 `WHERE id = 123` 或 `WHERE name LIKE '%王%'`）。
- **向量 DB**：將**語意轉化為多維空間中的幾何距離**。
  - `"機器學習"` 與 `"AI 模型訓練"` 雖然字面上沒有相同字眼，但在向量空間中距離極近。
  - 當使用者問「什麼是 AI？」，即使知識庫裡寫的是「人工智慧」，系統依然能藉由語意距離精確匹配。

---

### Q3. 我可以直接加入 PDF 文件嗎？
可以！
- 確保已安裝 `pip install pypdf`。
- 直接將 `.pdf` 文件丟入 `knowledge_base/` 資料夾中。
- 重新執行 `python 01_build_index.py`，程式會自動識別並逐頁切片加入索引。

---

### Q4. 為什麼基本 RAG 不能回答知識庫以外的問題？有了 API Key 難道不能自己上網查？
- **API Key 的本質**：只是取得 LLM（大腦）的運算推論額度，本身**並不具備上網即時瀏覽的能力**。
- **純 RAG 的限制**：Prompt 設定嚴格限制 LLM「僅能根據參考資料回答」，以防止模型瞎掰胡謅（Hallucination）。
- **解法（04_agent_search.py）**：引入 **Agent + Tools** 架構。賦予模型一把「搜尋引擎工具（DuckDuckGo）」，當模型發現知識庫沒有資料時，會自動調用搜尋引擎抓取最新網頁資訊。

---

### Q5. 專案分享與 Git 注意事項

1. **千萬不能推上 GitHub 的內容**：
   - ⚠️ `.env`：內含私人的 Google API Key，公開可能導致額度被盜用。
   - ⚠️ `chroma_db/`：二進位大型資料庫檔案，對方下載後只需自行跑一次 `01_build_index.py` 即可重建。
2. **Windows 網頁拖曳顯示「This file is hidden」問題**：
   - Windows 將 `.env.example`、`.gitignore` 等開頭帶點的檔案視為隱藏系統檔，GitHub 網頁拖曳會被阻擋。
   - **解決方式**：使用 Git 指令推送（`git add .` 會自動完整收錄這些檔案）。
3. **多人 / 公司帳號與個人帳號衝突**：
   - 若本地 Commit 被誤認成公司帳號，可在此專案目錄下指定本專案的專用作者：
     ```bash
     git config user.name "你的GitHub帳號"
     git config user.email "你的GitHub信箱"
     ```

---

## 🛠️ 常見錯誤與除錯指南

| 錯誤類型 | 原因分析 | 解決方式 |
|---------|---------|---------|
| `ModuleNotFoundError: No module named 'langchain.text_splitter'` | 新版 LangChain 將切片模組獨立 | 改為 `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| `UnicodeEncodeError: 'cp950' codec can't encode...` | Windows 終端機預設編碼不支援 Emoji | 在腳本頂部加入 `sys.stdout.reconfigure(encoding="utf-8")`，或在 CMD 輸入 `chcp 65001` |
| `404 NOT_FOUND: text-embedding-004 is not found` | Google 調整 API 支援名稱 | 使用可用模型別名 `models/gemini-embedding-001` |
| `429 RESOURCE_EXHAUSTED: Quota exceeded` | 免費方案的模型每日請求額度已滿（如 3.8-flash 每日限 20 次） | 將模型更換為額度更高的輕量版本（如 `gemini-3.5-flash-lite` 或 `gemini-flash-lite-latest`） |
| `ModuleNotFoundError: No module named 'ddgs'` | 新版 DuckDuckGo 套件名稱變更 | 執行 `pip install -U ddgs` 補齊依賴 |

---

## 🔄 換成其他向量資料庫（架構遷移）

LangChain 將向量資料庫封裝為統一口徑的 Retriever 介面。若未來要從 **ChromaDB** 遷移至生產環境資料庫，只需修改連線與初始化語法：

```
應用程式碼 (02_query / 03_advanced / 04_agent)
    ↓
LangChain 統一抽象介面 (.as_retriever() / .similarity_search())
    ↓
ChromaDB / FAISS / PostgreSQL (pgvector) / Qdrant (無痛切換)
```

- **FAISS**：Meta 出品，極速記憶體向量比對（適合中小型純 Python 部署）。
- **pgvector**：與 PostgreSQL 完美融合，兼顧關聯式商業資料與向量查詢（適合企業正式環境）。
- **Qdrant**：具備強大雲端叢集能力與視覺化管理介面（Web Dashboard）。
