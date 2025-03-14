import os
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM


# Step-1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Please set it in your environment variables.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ✅ Correct LLM Wrapper (Implements `_llm_type`)
class HFInferenceLLM(LLM):
    client: InferenceClient

    def _call(self, prompt: str, stop=None) -> str:
        """Generate text from the model using Hugging Face Inference API."""
        response = self.client.text_generation(prompt, max_new_tokens=512, temperature=0.3)
        return response.strip()

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

def load_llm(huggingface_repo_id):
    client = InferenceClient(model=huggingface_repo_id, token=HF_TOKEN)
    return HFInferenceLLM(client=client)

# Step-2: Connect LLM with FAISS and create chain
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

# Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ✅ Query Classification: Checks if question is medical-related
def is_medical_query_llm(query: str) -> bool:
    """
    Uses the LLM to classify if a query is medical-related.
    """
    prompt = f"""
    Classify the following question as 'medical' or 'non-medical'. 

    - A 'medical' question is related to diseases, symptoms, medications, healthcare, treatments, or medical research.
    - A 'non-medical' question is anything else.

    Only respond with one word: 'medical' or 'non-medical'.

    Query: {query}
    """

    response = load_llm(HUGGINGFACE_REPO_ID)._call(prompt).strip().lower()  
    print("DEBUG: Query Classification →", response)  # 🔍 Debugging classification

    return "medical" in response  # ✅ Fix: Ensures partial matches work



# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 7}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Run the query
user_query = input("Write Query here: ")

# ✅ First, check if the query is medical-related
if not is_medical_query_llm(user_query):
    print("I can only answer medical-related questions. Please ask about health or medicine.")
else:
    response = qa_chain.invoke({'query': user_query})
    print("\n💡 **RESULT:**", response["result"])
    print("\n📌 **SOURCE DOCUMENTS:**", response["source_documents"])
