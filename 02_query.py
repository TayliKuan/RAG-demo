"""
RAG Demo - 步驟 2：RAG 問答（查詢階段）
========================================
這個腳本負責：
1. 載入已建立的向量資料庫
2. 接收使用者問題
3. 從向量庫檢索相關文件（Retrieval）
4. 組合 Prompt 送給 Gemini 生成回答（Generation）

執行方式：
    python 02_query.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==================== 設定 ====================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ 請在 .env 檔案中設定 GOOGLE_API_KEY")

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"

# ==================== RAG Prompt 模板 ====================

RAG_PROMPT_TEMPLATE = """你是一個知識問答助手，請根據以下【參考資料】來回答使用者的問題。

【重要規則】
- 只能根據參考資料中的內容來回答
- 如果參考資料中沒有相關信息，請誠實說「我在知識庫中找不到相關資料」
- 回答請使用繁體中文
- 回答時請盡量清楚、有條理

【參考資料】
{context}

【使用者問題】
{question}

【回答】
"""

# ==================== 工具函數 ====================

def format_docs(docs):
    """將檢索到的文件格式化為字串"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知來源")
        formatted.append(f"[資料 {i}] 來源：{source}\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(vectorstore):
    """建立 RAG Chain"""

    # 檢索器：找出最相關的 4 個文件片段
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 使用餘弦相似度
        search_kwargs={"k": 4},    # 返回前 4 個最相關片段
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,  # 0=保守/確定, 1=創意/多樣
    )

    # Prompt 模板
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 組合 RAG Chain（LangChain LCEL 語法）
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ==================== 主流程 ====================

def main():
    print("=" * 50)
    print("🤖 RAG 問答系統")
    print("=" * 50)

    # ── 載入向量資料庫 ───────────────────────────────
    if not os.path.exists(CHROMA_DB_DIR):
        print("❌ 找不到向量資料庫！請先執行 01_build_index.py")
        return

    print("\n🔍 載入向量資料庫...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"   ✅ 向量資料庫載入完成")

    # ── 建立 RAG Chain ───────────────────────────────
    rag_chain, retriever = build_rag_chain(vectorstore)
    print("   ✅ RAG Chain 建立完成")
    print("\n💡 提示：輸入 'quit' 或 'exit' 離開\n")

    # ── 問答迴圈 ─────────────────────────────────────
    while True:
        print("-" * 50)
        question = input("❓ 請輸入問題：").strip()

        if not question:
            continue
        if question.lower() in ["quit", "exit", "離開", "結束"]:
            print("👋 再見！")
            break

        print("\n🔎 正在從知識庫檢索相關資料...")

        # 顯示檢索到的文件（透明化 RAG 過程）
        retrieved_docs = retriever.invoke(question)
        print(f"   找到 {len(retrieved_docs)} 個相關片段：")
        for i, doc in enumerate(retrieved_docs, 1):
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"   [{i}] {preview}...")

        print("\n🤖 Gemini 正在生成回答...")
        answer = rag_chain.invoke(question)

        print("\n📝 回答：")
        print(answer)
        print()


if __name__ == "__main__":
    main()
