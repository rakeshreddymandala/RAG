# 🚀 FAISS RAG-Based Medical Chatbot

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using **FAISS** for vector search and **Mistral-7B** as the LLM. The chatbot is designed to answer **medical-related queries** by retrieving relevant information from a **medical document database** and generating structured, informative responses.

## 🔥 Features
- **FAISS-Based Vector Search**: Efficient retrieval of medical documents.
- **Hugging Face Embeddings**: Uses `sentence-transformers/paraphrase-MiniLM-L3-v2` for vector embeddings.
- **LLM (Mistral-7B-Instruct-v0.3)**: Hosted on Hugging Face Inference API.
- **Custom Prompt Engineering**: Ensures structured, easy-to-read responses.
- **Medical Query Classification**: Filters out non-medical queries.
- **Streamlit UI**: Interactive chat-based interface.

## 🛠️ Tech Stack
- **Python 3.10+**
- **FAISS** (Facebook AI Similarity Search)
- **Hugging Face Transformers & Inference API**
- **LangChain** (for RAG implementation)
- **Streamlit** (for UI)

## ⚙️ Installation
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/faiss-rag-medical-chatbot.git
cd faiss-rag-medical-chatbot
```

### **2️⃣ Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate    # For Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Set Hugging Face API Token**
Get your Hugging Face API key from [Hugging Face](https://huggingface.co/settings/tokens) and set it as an environment variable:
```sh
export HF_TOKEN="your-huggingface-api-key"  # Mac/Linux
set HF_TOKEN="your-huggingface-api-key"  # Windows
```

## 🚀 Running the Chatbot
```sh
streamlit run app.py
```
This will start a local server. Open the provided **localhost URL** in your browser.

## 📜 Project Workflow
### **Phase 1: Create Memory**
✅ Load medical documents ➝ ✅ Chunking ➝ ✅ Create vector embeddings ➝ ✅ Store in FAISS

### **Phase 2: Connect Memory with LLM**
✅ Load FAISS ➝ ✅ Use Mistral-7B ➝ ✅ Implement RAG-based Retrieval

### **Phase 3: UI Development**
✅ Build Streamlit Chatbot UI ➝ ✅ Integrate RAG pipeline ➝ ✅ Deploy

## 🖥️ UI Preview
The chatbot interface is built with **Streamlit** and provides:
- ✅ A text input box for medical queries.
- ✅ AI-generated structured medical responses.
- ✅ Display of retrieved medical sources.

## 🏥 Example Query
**User:** "What are the symptoms of diabetes?"

**Chatbot Response:**
```
💡 **Answer:**
- Increased thirst and frequent urination
- Extreme hunger
- Unexplained weight loss
- Fatigue
- Blurred vision

⚠️ Please consult a doctor for medical advice.
```

## 📂 Project Structure
```
📦 faiss-rag-medical-chatbot
├── 📂 vectorstore/        # FAISS database
├── 📜 app.py              # Streamlit UI
├── 📜 rag_pipeline.py     # RAG implementation
├── 📜 requirements.txt    # Dependencies
├── 📜 README.md           # Project documentation
└── 📜 .gitignore          # Ignore unnecessary files
```

## 🛠️ Future Improvements
- ✅ Improve the **retrieval accuracy** with better embeddings.
- ✅ Add **more medical documents** to enhance AI knowledge.
- ✅ Deploy on **Hugging Face Spaces or AWS**.

## 💡 Credits
- **FAISS**: Meta AI
- **Mistral-7B**: Hugging Face
- **Streamlit**: Open-source UI framework

## 📜 License
This project is licensed under the **MIT License**.

---
_⭐ If you find this project useful, consider giving it a star on GitHub!_ 🌟

