"""
RAG Demo - 步驟 4：Agent + 搜尋工具（LangGraph 版）
=====================================================
結合兩種能力：
  1. ChromaDB 知識庫（你自己的文件）
  2. DuckDuckGo 網路搜尋（即時資料）

Agent 會自己判斷：
  - 問題在知識庫裡 → 用知識庫回答
  - 問題需要即時資料 → 自動上網搜尋

執行方式：
    python 04_agent_search.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ==================== 設定 ====================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("請在 .env 檔案中設定 GOOGLE_API_KEY")

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"


def build_agent():
    # ── 載入向量資料庫 ──────────────────────────────
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ── 工具 1：知識庫搜尋 ──────────────────────────
    @tool
    def knowledge_base_search(query: str) -> str:
        """搜尋本地知識庫。適用於 AI、機器學習、RAG、向量資料庫、台灣科技產業等主題。"""
        docs = retriever.invoke(query)
        if not docs:
            return "知識庫中找不到相關資料。"
        return "\n\n---\n\n".join(
            f"[來源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        )

    # ── 工具 2：DuckDuckGo 網路搜尋 ─────────────────
    web_search = DuckDuckGoSearchRun(
        name="web_search",
        description="搜尋網路上的即時資訊。適用於最新新聞、股價、天氣、即時事件等知識庫沒有的資訊。",
    )

    tools = [knowledge_base_search, web_search]

    # ── LLM ─────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    # ── LangGraph ReAct Agent ────────────────────────
    system_prompt = (
        "你是一個智慧問答助手，請用繁體中文回答。\n"
        "你有兩個工具可以使用：\n"
        "1. knowledge_base_search：搜尋本地知識庫（AI、RAG、機器學習等主題）\n"
        "2. web_search：搜尋網路即時資訊（新聞、股價、最新事件等）\n"
        "請根據問題性質選擇適合的工具，若知識庫就有答案就優先用知識庫。"
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
    return agent


def main():
    print("=" * 55)
    print("RAG + 網路搜尋 Agent")
    print("=" * 55)

    if not os.path.exists(CHROMA_DB_DIR):
        print("找不到向量資料庫！請先執行 01_build_index.py")
        return

    print("載入知識庫 + 初始化搜尋工具...")
    agent = build_agent()
    print("就緒！Agent 會自動選擇用知識庫或網路搜尋\n")
    print("範例問題：")
    print("  知識庫類：什麼是 RAG？")
    print("  即時資訊：最近 OpenAI 有什麼新消息？")
    print("  混合類  ：台積電在 AI 領域做了哪些事？")
    print("\n輸入 'quit' 離開\n")

    while True:
        print("-" * 55)
        question = input("問題：").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "離開"]:
            print("再見！")
            break

        print()
        try:
            print("思考中... (知識庫問題約 3 秒，網路搜尋約 10~30 秒，請稍等)")
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            # 取最後一則訊息（AI 的回答）
            last_msg = result["messages"][-1]
            content = last_msg.content

            # 新版 Gemini 回傳 list 格式，需要取出 text
            if isinstance(content, list):
                answer = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and "text" in item
                )
            else:
                answer = content

            print("回答：")
            print(answer)
        except Exception as e:
            print(f"發生錯誤：{e}")
        print()


if __name__ == "__main__":
    main()
