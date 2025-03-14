from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Step 1 - Load the document
data_path = "data/Mypdf.pdf"
loader = PyMuPDFLoader(data_path)
documents = loader.load()
full_text = "\n".join([doc.page_content for doc in documents])

print("Total pages loaded:", len(documents))
print("Final combined document length:", len(full_text))

# Step 2 - Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(extracted_data) if isinstance(extracted_data, str) else text_splitter.split_documents(extracted_data)

text_chunks = create_chunks(full_text)
print("Total chunks created:", len(text_chunks))
print("Chunk length:", len(text_chunks[0]))

# Step 3 - Create Vector Embeddings
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

def get_embeddings(text_chunks):
    return embedding_model.embed_documents(text_chunks) if isinstance(text_chunks[0], str) else embedding_model.embed_documents([chunk.page_content for chunk in text_chunks])

chunk_embeddings = get_embeddings(text_chunks)
print("Total embeddings created:", len(chunk_embeddings))
print("Embedding vector size:", len(chunk_embeddings[0]))

# Step 4 - Store the embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_texts(text_chunks, embedding_model)# ✅ Corrected
db.save_local(DB_FAISS_PATH)
print("FAISS database saved successfully!")
