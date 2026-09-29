from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import RagPipeline

app = FastAPI(
    title="RAG Q&A API",
    description=(
        "Ask questions over a small set of documents using a "
        "Retrieval-Augmented Generation pipeline (LangChain + FAISS)."
    ),
    version="1.0.0",
)

# Built once at startup. Rebuilding per-request would be wasteful; a larger
# system would persist the index instead of rebuilding it on every restart.
pipeline: RagPipeline | None = None


@app.on_event("startup")
def load_pipeline():
    global pipeline
    pipeline = RagPipeline()


class Question(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    excerpt: str


class Answer(BaseModel):
    answer: str
    mode: str  # "llm" if OPENAI_API_KEY is set, otherwise "extractive"
    sources: list[Source]


@app.post("/ask", response_model=Answer)
def ask_question(payload: Question):
    result = pipeline.ask(payload.question)
    return result


@app.get("/health")
def health_check():
    return {"status": "ok", "documents_loaded": pipeline is not None}
