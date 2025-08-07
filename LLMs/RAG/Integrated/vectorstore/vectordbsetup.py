import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import login

DATA_PATH = "ExternalKnowledge/"
DB_FAISS_PATH = "vectorstore/db_faiss"

HF_TOKEN = os.environ.get("HF_TOKEN")
login(HF_TOKEN)
# Step 1: Load raw PDF(s)
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store embeddings in FAISS
def create_vector_db(documents):
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db

# Load and create the VectorDB
documents = load_pdf_files(DATA_PATH)
vector_db = create_vector_db(documents)