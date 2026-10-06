
from pathlib import Path


# --- LLM Model Configuration ---
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_MAX_NEW_TOKENS: int = 450 # Controls how much new output the model can generate.
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_FREQUENCY_PENALTY: float = 0.2
LLM_PRESENCE_PENALTY: float = 0.2
#LLM_REPETITION_PENALTY: float = 1.03 # doesn't work for Groq
# LLM_QUESTION: str = "Which is the best football club in the world?"
LLM_SYSTEM_PROMPT: str = (
    "You are an ML Mentor who helps the user learn Machine Learning and "
    "Scikit-Learn. Use the information retrieved from the provided knowledge "
    "base as your source of knowledge. Do not use outside knowledge or the "
    "internet. You may create explanations, examples, quizzes, and exam-style "
    "questions based on the retrieved information. If the retrieved "
    "information does not contain enough information to answer the request, "
    "say that the information is not available in the knowledge base."
)
# CHAT_MEMORY_TOKEN_LIMIT: int = 2048  # Controls how much previous conversation is retained/provided as memory.

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "Tarka-AIR/Tarka-Embedding-150M-V1"


# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 3
# The size of each text chunk in tokens
CHUNK_SIZE: int = 1024
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 200


# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 1024 # Controls how much previous conversation is retained/provided as memory.


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"
