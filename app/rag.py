"""
Small RAG (Retrieval-Augmented Generation) pipeline.

Design notes:
- Uses LangChain's document loaders/splitters and its FAISS vector store
  wrapper, so the pipeline follows the same shape you'd use with any
  embeddings provider.
- Embeddings are TF-IDF vectors (scikit-learn) rather than a downloaded
  transformer model. This keeps the project runnable offline with no API
  keys or model downloads — useful for a portfolio project people can clone
  and run immediately. Swapping in HuggingFaceEmbeddings or OpenAIEmbeddings
  is a one-class change (see `TfidfEmbeddings` below for where it plugs in).
- If OPENAI_API_KEY is set, retrieved chunks are passed to an LLM to produce
  a natural-language answer. Otherwise, the pipeline returns the best-
  matching passage directly (extractive mode), so the app is still useful
  with zero external dependencies at runtime.
"""

import glob
import os
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import DOCUMENTS_DIR, OPENAI_API_KEY, OPENAI_MODEL, TOP_K


class TfidfEmbeddings(Embeddings):
    """A LangChain-compatible Embeddings class backed by scikit-learn TF-IDF.

    Fit once on the document corpus, then used to embed both the stored
    chunks and incoming queries into the same vector space.
    """

    def __init__(self, corpus: List[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(corpus)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()


class RagPipeline:
    def __init__(self, documents_dir: str = DOCUMENTS_DIR, top_k: int = TOP_K):
        self.top_k = top_k
        self.vectorstore = self._build_index(documents_dir)
        self._llm = self._build_llm()

    def _load_documents(self, documents_dir: str) -> List[Document]:
        docs = []
        for path in sorted(glob.glob(os.path.join(documents_dir, "*.txt"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))
        if not docs:
            raise RuntimeError(
                f"No .txt documents found in '{documents_dir}'. Add some source "
                "files before starting the API."
            )
        return docs

    def _build_index(self, documents_dir: str) -> FAISS:
        raw_docs = self._load_documents(documents_dir)

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(raw_docs)

        corpus = [c.page_content for c in chunks]
        embeddings = TfidfEmbeddings(corpus)

        return FAISS.from_documents(chunks, embeddings)

    def _build_llm(self):
        if not OPENAI_API_KEY:
            return None
        # Imported lazily so the app runs without langchain-openai installed
        # correctly configured only when someone actually wants LLM answers.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    def ask(self, question: str) -> dict:
        results = self.vectorstore.similarity_search(question, k=self.top_k)
        sources = [
            {"source": r.metadata.get("source", "unknown"), "excerpt": r.page_content}
            for r in results
        ]

        if self._llm is not None:
            context = "\n\n".join(r.page_content for r in results)
            prompt = (
                "Answer the question using only the context below. "
                "If the answer isn't in the context, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            response = self._llm.invoke(prompt)
            answer = response.content
            mode = "llm"
        else:
            # Extractive fallback: no LLM configured, so return the single
            # most relevant passage as the "answer".
            answer = results[0].page_content if results else "No relevant passage found."
            mode = "extractive"

        return {"answer": answer, "mode": mode, "sources": sources}
