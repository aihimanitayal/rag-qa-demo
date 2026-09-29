import os

# Optional: if set, the app uses an OpenAI chat model to generate a natural-
# language answer from the retrieved context. If not set, the app falls back
# to a free, fully local extractive mode (returns the most relevant passage
# directly), so the project runs with zero paid API keys out of the box.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")

# How many chunks to retrieve per question.
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
