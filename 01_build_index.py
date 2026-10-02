"""
RAG Demo - 步驟 1：建立向量資料庫（索引階段）
================================================
這個腳本負責：
1. 讀取知識庫文件（支援 .md 和 .pdf）
2. 將文件切片（Chunking）
3. 轉換成向量（Embedding）
4. 儲存至 ChromaDB

執行方式：
    python 01_build_index.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ==================== 設定 ====================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("請在 .env 檔案中設定 GOOGLE_API_KEY")

KNOWLEDGE_BASE_DIR = "./knowledge_base"   # 知識庫資料夾
CHROMA_DB_DIR = "./chroma_db"             # ChromaDB 儲存位置
COLLECTION_NAME = "rag_demo"              # 集合名稱

# ==================== 主流程 ====================

def load_documents():
    """讀取 knowledge_base 裡的 .md 和 .pdf 檔案"""
    all_docs = []

    # 讀取 Markdown / txt 文件
    md_loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    md_docs = md_loader.load()
    all_docs.extend(md_docs)
    print(f"   Markdown: {len(md_docs)} 份")

    # 讀取 PDF 文件（需要安裝 pypdf）
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        pdf_loader = PyPDFDirectoryLoader(KNOWLEDGE_BASE_DIR)
        pdf_docs = pdf_loader.load()
        if pdf_docs:
            all_docs.extend(pdf_docs)
            print(f"   PDF: {len(pdf_docs)} 頁")
        else:
            print(f"   PDF: 0 份（資料夾內無 PDF 檔案）")
    except ImportError:
        print(f"   PDF 讀取略過（請執行 pip install pypdf 來支援 PDF）")
    except Exception as e:
        print(f"   PDF 讀取略過（{e}）")

    return all_docs


def main():
    print("=" * 50)
    print("RAG 索引建立程式")
    print("=" * 50)

    # 步驟 1：讀取文件
    print("\n步驟 1：讀取知識庫文件...")
    documents = load_documents()
    print(f"   共讀取 {len(documents)} 份文件/頁")
    for doc in documents:
        print(f"      - {doc.metadata.get('source', 'unknown')}")

    if not documents:
        print("錯誤：knowledge_base 資料夾內沒有文件！")
        return

    # 步驟 2：文件切片（Chunking）
    print("\n步驟 2：文件切片（Chunking）...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # 每片最多 500 字
        chunk_overlap=50,     # 相鄰片段重疊 50 字（保留上下文）
        separators=["\n---\n", "\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   切成 {len(chunks)} 個文字片段（chunks）")
    print(f"   平均每片約 {sum(len(c.page_content) for c in chunks) // len(chunks)} 字")

    # 步驟 3：初始化 Embedding 模型
    print("\n步驟 3：初始化 Gemini Embedding 模型...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    print("   Embedding 模型載入完成")

    # 步驟 4：儲存至 ChromaDB
    print("\n步驟 4：將 Chunks 轉換為向量並儲存至 ChromaDB...")
    print(f"   儲存位置：{CHROMA_DB_DIR}")

    # 如果已存在舊的資料庫，先清除
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        shutil.rmtree(CHROMA_DB_DIR)
        print("   已清除舊的向量資料庫")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    print(f"   成功儲存 {len(chunks)} 個向量！")
    print(f"\n索引建立完成！請執行 02_query.py 開始問答")
    print("=" * 50)


if __name__ == "__main__":
    main()
