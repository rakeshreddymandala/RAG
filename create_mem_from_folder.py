import os
import multiprocessing
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
DATA_FOLDER = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load Embedding Model
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# Load existing FAISS index (if it exists)
if os.path.exists(DB_FAISS_PATH):
    faiss_index = FAISS.load_local(DB_FAISS_PATH, embedding_model)
    processed_pdfs = set(doc.metadata["source"] for doc in faiss_index.similarity_search("", k=1000))
else:
    faiss_index = None
    processed_pdfs = set()

# Function to process a single PDF
def process_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    
    if pdf_name in processed_pdfs:
        print(f"Skipping already processed PDF: {pdf_name}")
        return []
    
    try:
        loader = UnstructuredPDFLoader(pdf_path)  
        documents = loader.load()  

        for doc in documents:
            doc.metadata["source"] = pdf_name  

        return documents
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []

# Get all PDFs in the folder
pdf_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

# Step 1: Load all PDFs using parallel processing (only new ones)
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    all_documents = pool.map(process_pdf, pdf_files)

# Flatten the list
documents = [doc for sublist in all_documents for doc in sublist]

if documents:
    # Step 2: Create Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # Step 3: Create Vector Embeddings
    chunk_texts = [chunk.page_content for chunk in text_chunks]
    chunk_embeddings = embedding_model.embed_documents(chunk_texts)

    # Step 4: FAISS Indexing for Faster Search
    if faiss_index:
        faiss_index.add_texts(text_chunks)
    else:
        faiss_index = FAISS.from_texts(text_chunks, embedding_model)

    # Save FAISS DB
    faiss_index.save_local(DB_FAISS_PATH)
    print("FAISS database updated successfully!")
else:
    print("No new PDFs to process.")
