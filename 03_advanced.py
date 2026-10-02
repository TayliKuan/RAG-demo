"""
RAG Demo - 進階版：加入 PDF 支援 + 評分顯示
============================================
在基礎版之上新增：
- 📄 支援讀取 PDF 文件
- 📊 顯示每個檢索結果的相似度分數
- 🔄 支援多輪對話（記憶上下文）

執行方式：
    python 03_advanced.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"

# 帶歷史記憶的 Prompt
CHAT_RAG_PROMPT = """你是一個知識問答助手。請根據【參考資料】和【對話歷史】來回答問題。

【參考資料】
{context}

【重要規則】
- 優先使用參考資料中的內容
- 如找不到相關資料，請誠實告知
- 使用繁體中文回答
"""

def main():
    print("=" * 55)
    print("🚀 RAG 進階問答系統（支援多輪對話）")
    print("=" * 55)

    if not os.path.exists(CHROMA_DB_DIR):
        print("❌ 找不到向量資料庫！請先執行 01_build_index.py")
        return

    # 載入向量資料庫
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    chat_history = []  # 對話歷史
    print("\n✅ 系統就緒！（輸入 'quit' 離開，'clear' 清除對話歷史）\n")

    while True:
        print("-" * 55)
        question = input("❓ 問題：").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit"]:
            print("👋 再見！")
            break
        if question.lower() == "clear":
            chat_history = []
            print("🗑️  對話歷史已清除")
            continue

        # 檢索（使用分數篩選低品質結果）
        results = vectorstore.similarity_search_with_relevance_scores(question, k=4)

        print(f"\n🔎 檢索結果（相似度分數）：")
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            preview = doc.page_content[:50].replace("\n", " ")
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"   [{i}] {bar} {score:.2f} | {preview}...")
            if score > 0.3:  # 只使用相似度 > 0.3 的結果
                context_parts.append(doc.page_content)

        context = "\n\n".join(context_parts) if context_parts else "（無高度相關資料）"

        # 組合帶歷史的 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CHAT_RAG_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()

        print("\n🤖 回答：")
        answer = chain.invoke({
            "context": context,
            "history": chat_history,
            "question": question,
        })
        print(answer)

        # 更新對話歷史
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # 保留最近 6 輪對話（避免 Token 超出）
        if len(chat_history) > 12:
            chat_history = chat_history[-12:]

        print(f"\n💬 對話輪數：{len(chat_history) // 2}")


if __name__ == "__main__":
    main()
