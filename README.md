# RAG Q&A API

A small FastAPI service that answers questions over a set of local documents using a Retrieval-Augmented Generation (RAG) pipeline built with LangChain and FAISS.

Ask a question like *"How many days of leave do employees get?"* and the API searches the document set, retrieves the most relevant passages, and returns an answer with source citations.

## How it works

1. **Load & chunk** — documents in `documents/` are loaded and split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
2. **Embed & index** — each chunk is embedded into a vector using TF-IDF (scikit-learn) and stored in a FAISS vector index.
3. **Retrieve** — an incoming question is embedded the same way, and FAISS returns the most similar chunks.
4. **Answer** — if `OPENAI_API_KEY` is set, the retrieved chunks are passed to an LLM to generate a natural-language answer. If not, the API returns the most relevant passage directly (extractive mode) — so the project runs with **zero paid API keys or model downloads**.

This keeps the project runnable offline out of the box, while the pipeline shape (loaders → splitter → embeddings → vector store → retrieval → LLM) is the same one used with any embeddings provider — swapping in OpenAI or HuggingFace embeddings is a one-class change in `app/rag.py`.

## Tech Stack

Python · FastAPI · LangChain · FAISS · scikit-learn · (optional) OpenAI

## Project Structure

```
rag-qa-api/
├── app/
│   ├── main.py       # FastAPI app + /ask endpoint
│   ├── rag.py         # RAG pipeline: loading, chunking, indexing, retrieval, answer generation
│   └── config.py      # Environment-based configuration
├── documents/          # Source .txt files the API answers questions about
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Running Locally

1. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. (Optional) Copy `.env.example` to `.env` and add an `OPENAI_API_KEY` if you want LLM-generated answers instead of extractive passages.
3. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Open `http://localhost:8000/docs` to try it interactively.

## Running with Docker

```bash
docker build -t rag-qa-api .
docker run -p 8000:8000 --env-file .env rag-qa-api
```

## Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of leave do employees accrue per month?"}'
```

Response (extractive mode, no API key set):
```json
{
  "answer": "Employees accrue 1.5 days of paid leave per month, capped at 24 days per year...",
  "mode": "extractive",
  "sources": [
    {"source": "company_handbook.txt", "excerpt": "..."}
  ]
}
```

## Adding Your Own Documents

Drop any `.txt` files into `documents/` and restart the API — they'll be automatically loaded, chunked, and indexed on startup.

## Notes

This project was built to get hands-on practice with the core RAG pattern (chunking, embeddings, vector search, retrieval-augmented answer generation) as a follow-up to production API work with FastAPI, PostgreSQL, and third-party integrations.
