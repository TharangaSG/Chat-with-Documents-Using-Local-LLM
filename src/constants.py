EMBEDDING_MODEL_PATH = "embedding_model"  # OR Path of local eg. "embedding_model/"" or the name of SentenceTransformer model eg. "sentence-transformers/all-mpnet-base-v2" from Hugging Face
ASSYMETRIC_EMBEDDING = False  
EMBEDDING_DIMENSION = 768  
TEXT_CHUNK_SIZE = 300  

OLLAMA_MODEL_NAME = (
    "llama3.2:1b"  
)

# Logging
LOG_FILE_PATH = "logs/app.log" 
# OpenSearch settings
OPENSEARCH_HOST = "localhost"  
OPENSEARCH_PORT = 9200  
OPENSEARCH_INDEX = "documents"  