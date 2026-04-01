import os
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

_active_vectordb = None


def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


def build_vectordb(filepath: str) -> int:
    global _active_vectordb

    loader = PyPDFLoader(filepath)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embedding_model()

    if _active_vectordb is not None:
        _active_vectordb.delete_collection()

    _active_vectordb = Chroma.from_documents(chunks, embeddings)
    return len(chunks)


def answer_query(query: str) -> str:
    if _active_vectordb is None:
        raise ValueError("No document has been uploaded yet.")

    docs = _active_vectordb.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    model = genai.GenerativeModel("gemini-3-flash-preview")
    prompt = f"""You are a helpful assistant. Answer the question based only on the context provided below.
If the answer is not in the context, say "I could not find the answer in the provided document."

Context:
{context}

Question: {query}

Answer:"""

    response = model.generate_content(prompt)
    return response.text