import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
import nest_asyncio

# Apply Nest Asyncio to avoid event loop issues
nest_asyncio.apply()

# ✅ Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

# ✅ Load Hugging Face Token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN is not set. Please set it in your environment variables.")
    st.stop()

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ✅ LLM Wrapper with Error Handling
class HFInferenceLLM(LLM):
    client: InferenceClient

    def _call(self, prompt: str, stop=None) -> str:
        try:
            response = self.client.text_generation(prompt, max_new_tokens=512, temperature=0.3)
            return response.strip()
        except Exception as e:
            return f"⚠️ Error generating response: {e}"

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

# ✅ Load LLM and FAISS with Streamlit Cache
@st.cache_resource
def load_llm():
    client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)
    return HFInferenceLLM(client=client)

@st.cache_resource
def load_faiss():
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"⚠️ FAISS database not found at {DB_FAISS_PATH}. Please generate embeddings first.")
        st.stop()
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load FAISS & LLM
db = load_faiss()
llm = load_llm()

# ✅ Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
You are an AI medical assistant. Use the provided context to generate a structured, informative answer.

📌 **Guidelines:**
- Provide a **clear and structured** answer.
- Use **bullet points** and **bold headings** where needed.
- Keep it **easy to read** for non-medical users.
- End with a **medical disclaimer**.

📌 **Context:**
{context}

📌 **User’s Question:**
{question}

💡 **AI's Answer:**
"""

def custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ✅ Query Classification: Checks if question is medical-related
def is_medical_query(query: str) -> bool:
    classification_prompt = f"""
    Classify the following question as 'medical' or 'non-medical'. 

    - A 'medical' question is related to diseases, symptoms, medications, healthcare, treatments, or medical research.
    - A 'non-medical' question is anything else.

    Only respond with one word: 'medical' or 'non-medical'.

    Query: {query}
    """
    
    response = llm._call(classification_prompt).strip().lower()
    return "medical" in response

# ✅ Create QA Chain with FAISS Retrieval (Optimized `k=5`)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# ✅ Streamlit UI Layout
st.title("MediBot 🏥")
st.write("💬 Ask me any medical-related question!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Pass Your prompt here:")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # ✅ Check if query is medical-related
    if not is_medical_query(prompt):
        response = "❌ I can only answer **medical-related** questions. Please ask about health or medicine."
    else:
        retrieved_docs = db.as_retriever(search_kwargs={'k': 5}).get_relevant_documents(prompt)

        if not retrieved_docs:
            response = "⚠️ No relevant medical information found. Try rephrasing your question."
        else:
            result = qa_chain.invoke({'query': prompt})
            response = f"💡 **Answer:**\n{result['result']}"

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
