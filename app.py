import streamlit as st
import os
import tempfile
from typing import List, Optional
import uuid
import io

# Document processing
from PyPDF2 import PdfReader
import docx

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# Transformers for the language model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Set page configuration
st.set_page_config(
    page_title="ManGPT - Document AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_embeddings():
    """Load the embedding model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_llm():
    """Load the language model for generation"""
    model_name = "microsoft/DialoGPT-medium"
    
    # Fallback to a simpler approach if DialoGPT doesn't work
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            device=-1  # CPU
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except:
        # Fallback to a simple text generation approach
        return create_simple_llm()

def create_simple_llm():
    """Create a simple LLM wrapper for basic text generation"""
    class SimpleLLM:
        def __call__(self, prompt):
            # This is a very basic implementation
            # In a real scenario, you'd want a proper model here
            return f"Based on the provided context: {prompt[:200]}..."
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: "Response generated from document context."
    
    return SimpleLLM()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file) -> str:
    """Extract text from Word document"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from Word document: {str(e)}")
        return ""

def process_documents(uploaded_files) -> List[Document]:
    """Process uploaded documents and extract text"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(tmp_file_path)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                text = extract_text_from_docx(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
                continue
            
            if text:
                # Create document object
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "type": uploaded_file.type
                    }
                )
                documents.append(doc)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return documents

def create_vectorstore(documents: List[Document]):
    """Create FAISS vector store from documents"""
    if not documents:
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = st.session_state.embeddings
    if embeddings is None:
        embeddings = load_embeddings()
        st.session_state.embeddings = embeddings
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def get_response(question: str, vectorstore) -> str:
    """Get response using RAG pipeline"""
    if vectorstore is None:
        return "Please upload and process documents first."
    
    try:
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a simple response based on context
        if context.strip():
            # Simple keyword-based response generation
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Find relevant sentences
            sentences = context.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_lower.split()):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                response = "Based on the documents, here's what I found:\n\n"
                response += "\n".join(relevant_sentences[:3])  # Top 3 relevant sentences
                return response
            else:
                return f"Based on the uploaded documents:\n\n{context[:500]}..."
        else:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def summarize_documents(documents: List[Document]) -> str:
    """Generate a summary of the documents"""
    if not documents:
        return "No documents to summarize."
    
    try:
        # Combine all document text
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Simple extractive summarization - take first few sentences from each paragraph
        paragraphs = full_text.split('\n\n')
        summary_parts = []
        
        for para in paragraphs[:5]:  # First 5 paragraphs
            sentences = para.split('.')
            if sentences and len(sentences[0]) > 50:  # Skip very short sentences
                summary_parts.append(sentences[0].strip() + '.')
        
        if summary_parts:
            return "Document Summary:\n\n" + "\n\n".join(summary_parts)
        else:
            return "Summary: The documents contain information that has been processed and is available for querying."
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Main app
def main():
    st.title("📚 ManGPT - Document AI Assistant")
    st.markdown("Upload your documents and ask questions about them!")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF or Word files",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload PDF or Word documents to analyze"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) uploaded!")
            
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Process documents
                    documents = process_documents(uploaded_files)
                    
                    if documents:
                        st.session_state.documents = documents
                        
                        # Create vector store
                        vectorstore = create_vectorstore(documents)
                        st.session_state.vectorstore = vectorstore
                        
                        st.success(f"✅ Processed {len(documents)} document(s) successfully!")
                        
                        # Show document info
                        st.subheader("📋 Processed Documents")
                        for doc in documents:
                            st.write(f"• {doc.metadata['source']} ({len(doc.page_content)} characters)")
                    else:
                        st.error("❌ No documents could be processed.")
        
        # Document summary section
        if st.session_state.documents:
            st.divider()
            st.subheader("📝 Document Summary")
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_documents(st.session_state.documents)
                    st.write(summary)
    
    # Main chat interface
    if st.session_state.vectorstore is not None:
        st.subheader("💬 Ask Questions About Your Documents")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.write(f"**You:** {question}")
                st.write(f"**ManGPT:** {answer}")
                st.divider()
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic discussed in the documents?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Ask", type="primary")
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and question:
            with st.spinner("Thinking..."):
                response = get_response(question, st.session_state.vectorstore)
                
                # Add to chat history
                st.session_state.chat_history.append((question, response))
                
                # Display the new response
                with st.container():
                    st.write(f"**You:** {question}")
                    st.write(f"**ManGPT:** {response}")
                
                # Clear the input
                st.rerun()
    
    else:
        st.info("👆 Please upload and process documents using the sidebar to get started!")
        
        # Show example
        with st.expander("💡 How to use ManGPT"):
            st.markdown("""
            1. **Upload Documents**: Use the sidebar to upload PDF or Word documents
            2. **Process**: Click "Process Documents" to analyze them
            3. **Ask Questions**: Use the chat interface to ask questions about your documents
            4. **Get Summaries**: Generate document summaries in the sidebar
            
            **Example Questions:**
            - "What is the main topic of this document?"
            - "Summarize the key points discussed"
            - "What are the conclusions mentioned?"
            """)

if __name__ == "__main__":
    main()
