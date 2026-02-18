import os
import shutil
from typing import List

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


from ba_ragmas_chatbot.paths import DB_DIR


DB_DIR_STR = str(DB_DIR)


def get_embedding_function():
    """defines embedding-model!"""
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434",
    )


def setup_vectorstore(documents_paths: List[str]):
    """
    deletes old db and creates it new with current documents.
    """

    if os.path.exists(DB_DIR_STR):
        shutil.rmtree(DB_DIR_STR)

    if not documents_paths:
        print("ℹ️ no documents to index.")
        return None

    docs = []

    for path in documents_paths:
        try:
            if path.startswith("http://") or path.startswith("https://"):
                print(f"🌐 Loading URL content: {path}")
                loader = WebBaseLoader(path)
            elif path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".docx"):
                loader = Docx2txtLoader(path)
            elif path.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                continue

            docs.extend(loader.load())
            print(f"✅ document loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"⚠️ error loading {path}: {e}")

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)

    os.makedirs(DB_DIR_STR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=get_embedding_function(),
        persist_directory=DB_DIR_STR,
    )

    print(f"💾 Vectorstore created in {DB_DIR_STR} with {len(splits)} chunks.")
    return vectorstore


def get_retriever(k: int = 4):
    """hands back retriever for agents."""
    if not os.path.exists(DB_DIR_STR):
        return None

    vectorstore = Chroma(
        persist_directory=DB_DIR_STR,
        embedding_function=get_embedding_function(),
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})
