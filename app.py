# app.py
import streamlit as st
import os, tempfile
from io import BytesIO

# PDF & Word parsing
from PyPDF2 import PdfReader
import docx

# LangChain & embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Hugging Face model pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------------------
# Config
# --------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-small"
CHROMA_PERSIST_DIR = "chroma_man_gpt_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --------------------
# File parsing helpers
# --------------------
def read_pdf(file_bytes: BytesIO) -> str:
    reader = PdfReader(file_bytes)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

def read_docx(file_bytes: BytesIO) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes.read()); tmp.flush()
        doc = docx.Document(tmp.name)
    os.remove(tmp.name)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif name.endswith(".docx") or name.endswith(".doc"):
        uploaded_file.seek(0); return read_docx(uploaded_file)
    else:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

# --------------------
# Models
# --------------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_generation_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
    hf_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return HuggingFacePipeline(pipeline=hf_pipe)

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_vectordb(texts, metadatas):
    emb = get_embedding_model()
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    vectordb = Chroma.from_documents(docs, emb, persist_directory=CHROMA_PERSIST_DIR)
    vectordb.persist()
    return vectordb

# --------------------
# Streamlit UI
# --------------------
def main():
    st.title("📘 ManGPT — Document Q&A")
    st.markdown("Upload a PDF or Word document, index it, and ask questions.")

    uploaded = st.file_uploader("Upload a PDF or Word (.docx)", type=["pdf", "docx"])
    if "vectordb" not in st.session_state: st.session_state.vectordb = None

    if uploaded:
        raw_text = load_file(uploaded)
        st.info(f"Extracted {len(raw_text)} characters from document")
        texts = split_text(raw_text)
        metadatas = [{"source": uploaded.name, "chunk": i} for i in range(len(texts))]

        if st.button("Index document"):
            st.session_state.vectordb = create_vectordb(texts, metadatas)
            st.success("Document indexed!")

    if st.session_state.vectordb:
        q = st.text_input("Ask a question about your document")
        if st.button("Get Answer"):
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
            llm = get_generation_pipeline()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            result = qa_chain({"query": q})
            st.write("**Answer:**", result["result"])

        if st.button("Summarize Document"):
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 6})
            docs = retriever.get_relevant_documents("summarize this document")
            combined = "\n".join([d.page_content for d in docs])
            prompt = f"Summarize this:\n\n{combined}"
            llm = get_generation_pipeline()
            summary = llm(prompt)
            st.write("**Summary:**", summary)

if __name__ == "__main__":
    main()
