# RAG Demo - 知識庫文件範例

## 什麼是人工智慧？

人工智慧（Artificial Intelligence，AI）是指由電腦系統展現的智慧，模擬人類的學習、推理、問題解決等能力。
AI 主要分為以下幾類：
- **弱人工智慧（Narrow AI）**：專注於特定任務，如語音辨識、圖像識別
- **強人工智慧（General AI）**：能夠執行任何人類能做的智慧任務（目前仍是理論）
- **超人工智慧（Super AI）**：超越人類智慧的 AI（目前仍是假設）

---

## 什麼是機器學習？

機器學習（Machine Learning，ML）是 AI 的子領域，讓電腦系統從數據中自動學習和改進，而無需明確編程。

主要分類：
1. **監督式學習**：使用標記數據訓練模型（例：垃圾郵件分類）
2. **非監督式學習**：從未標記數據中找出規律（例：客戶分群）
3. **強化學習**：透過獎懲機制讓模型學習最佳策略（例：AlphaGo）

---

## 什麼是深度學習？

深度學習（Deep Learning）是機器學習的子集，使用多層神經網路（Neural Network）來學習數據的複雜模式。

深度學習的突破應用：
- **ChatGPT / Gemini**：大型語言模型（LLM）
- **Stable Diffusion / DALL-E**：圖像生成
- **Whisper**：語音轉文字
- **AlphaFold**：蛋白質結構預測

---

## 什麼是 RAG？

RAG（Retrieval-Augmented Generation，檢索增強生成）是一種 AI 技術，結合了：
1. **檢索（Retrieval）**：從知識庫找出相關文件
2. **生成（Generation）**：用 LLM 根據檢索結果生成回答

RAG 的優點：
- 解決 LLM 知識截止日期問題
- 減少幻覺（Hallucination）
- 可使用私有數據
- 回答可追溯來源

RAG 的運作流程：
1. 將文件切片（Chunking）
2. 將文件轉為向量（Embedding）
3. 儲存至向量資料庫（Vector DB）
4. 使用者提問時，先從 Vector DB 檢索相關文件
5. 將問題 + 相關文件一起送給 LLM 生成回答

---

## 台灣的 AI 發展

台灣在 AI 領域有重要貢獻：
- **台積電（TSMC）**：生產全球最先進的 AI 晶片
- **聯發科（MediaTek）**：開發邊緣 AI 晶片
- **鴻海（Foxconn）**：AI 工廠與智慧製造
- **研華（Advantech）**：工業 AI 解決方案

台灣政府在 2023 年宣布「AI 台灣行動計畫」，投入大量資源發展 AI 人才與產業。

---

## 向量資料庫比較

| 資料庫 | 類型 | 特點 | 適合場景 |
|--------|------|------|----------|
| ChromaDB | 輕量本地 | 零設定、純 Python | Demo、小型專案 |
| FAISS | 本地函式庫 | Meta 出品、速度快 | 研究、中型專案 |
| Pinecone | 雲端服務 | 全託管、易擴展 | 生產環境 |
| pgvector | PostgreSQL 擴充 | 結合傳統 DB | 已有 PG 基礎架構 |
| Weaviate | 雲端/本地 | 功能豐富 | 企業應用 |
