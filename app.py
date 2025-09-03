
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain  # ✅ fixed import
from transformers import pipeline

# ------------------------------
# Streamlit App Config
# ------------------------------
st.set_page_config(page_title="ManGPT", page_icon="🤖", layout="wide")
st.title("🤖 ManGPT – Document Q&A & Summarization")
st.markdown("Upload PDFs or Word documents, then **ask questions** or get a **summary**.")

# ------------------------------
# Initialize Session State Safely
# ------------------------------
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None

# ------------------------------
# Load Document Function
# ------------------------------
def load_document(file):
    """Load PDF or Word document and return text."""
    if file.type == "application/pdf":
        loader = PyPDFLoader(file.name)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file.name)
    else:
        st.error("❌ Unsupported file type. Please upload PDF or Word.")
        return None
    return loader.load()

# ------------------------------
# Split Text Function
# ------------------------------
def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# ------------------------------
# Create FAISS VectorDB
# ------------------------------
def create_vectordb(texts, metadatas=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# ------------------------------
# LLM Setup (CPU-friendly)
# ------------------------------
def load_llm():
    gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",  # lightweight & HF-friendly
        tokenizer="google/flan-t5-small",
        max_length=512
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)

llm = load_llm()

# ------------------------------
# File Uploader
# ------------------------------
uploaded = st.file_uploader("📤 Upload a PDF or Word document", type=["pdf", "docx"])

if uploaded:
    with open(uploaded.name, "wb") as f:
        f.write(uploaded.getbuffer())
    docs = load_document(uploaded)

    if docs:
        texts = split_text(docs)

        if st.button("📥 Index Document"):
            st.session_state["vectordb"] = create_vectordb(texts)
            st.success("✅ Document indexed successfully!")

# ------------------------------
# Q&A Section
# ------------------------------
if st.session_state.get("vectordb") is not None:
    q = st.text_input("💬 Ask a question about your document:")

    if st.button("🔍 Get Answer") and q.strip():
        retriever = st.session_state["vectordb"].as_retriever(search_kwargs={"k": 4})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa_chain.invoke(q)

        st.subheader("Answer:")
        st.write(result["result"])

# ------------------------------
# Summarization Section
# ------------------------------
if st.session_state.get("vectordb") is not None:
    if st.button("📝 Summarize Document"):
        docs = st.session_state["vectordb"].docstore.search("")
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)

        st.subheader("Summary:")
        st.write(summary)
