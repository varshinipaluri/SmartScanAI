import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration
DB_DIR = "./chroma_db"
UPLOAD_DIR = "./uploaded_files"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_FILE_SIZE_MB = 10
MAX_URL_CONTENT_LENGTH = 50000
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="SmartScan AI - Smart Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for better UI
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: white;
        color: black;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: gray-600;
        transform: scale(1.02);
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f1f1f1;
        border-left: 4px solid #4CAF50;
    }
    .stSpinner>div>div {
        border-color: #4CAF50 transparent transparent transparent;
    }
    .file-info {
        padding: 0.5rem;
        background: #f0f2f6;
        border-radius: 5px;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    .tab-container {
        margin-top: 1rem;
    }
    .url-input {
        margin-bottom: 1rem;
    }
    .source-container {
        margin-top: 10px;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .source-item {
        margin: 5px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pdf-source {
        color: #d93025;
        font-weight: 500;
    }
    .url-source {
        color: #1a73e8;
        text-decoration: none;
        font-weight: 500;
    }
    .download-btn {
        background: #4CAF50;
        color: white;
        border: none;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced prompt template
prompt_template = """
You are SmartScan, an expert at analyzing documents and web content. Follow these guidelines:

1. Provide comprehensive, accurate answers based on the context
2. If information isn't available, say "I couldn't find this in the provided sources"
3. Maintain a professional yet approachable tone
4. When relevant, reference the document section your answer comes from

Context: {context}
Question: {question}

Provide a detailed, well-structured answer:
"""

class DocumentProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_db = None
        self.source_files = {}  # Store original files for download
        self.source_urls = {}   # Store URLs
        self.qa_prompt = PromptTemplate.from_template(prompt_template)

    def process_pdf(self, file):
        try:
            # Save the original file for download
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            self.source_files[file.name] = {
                "path": file_path,
                "file_obj": file
            }
            
            # Extract text
            text = ""
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            if not text.strip():
                st.warning(f"File {file.name} has no extractable text")
                return None
                
            return text
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return None

    def process_url(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer']):
                element.decompose()
                
            text = ' '.join([
                element.get_text() 
                for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'article'])
            ])
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                st.warning(f"No extractable content from {url}")
                return None
                
            self.source_urls[url] = {
                "domain": urlparse(url).netloc
            }
            
            return text[:MAX_URL_CONTENT_LENGTH]
        except Exception as e:
            st.error(f"Error processing {url}: {str(e)}")
            return None

    def process_sources(self, pdf_files=None, urls=None):
        """Process both PDFs and URLs, return number of chunks created"""
        all_texts = []
        
        # Process PDFs
        if pdf_files:
            for file in pdf_files:
                text = self.process_pdf(file)
                if text:
                    all_texts.append((text, file.name))
        
        # Process URLs
        if urls:
            for url in urls:
                text = self.process_url(url)
                if text:
                    all_texts.append((text, url))
        
        if not all_texts:
            return 0
        
        # Create vector database
        chunks = []
        for text, source in all_texts:
            text_chunks = self.text_splitter.split_text(text)
            chunks.extend([
                Document(page_content=chunk, metadata={"source": source}) 
                for chunk in text_chunks
            ])
        
        try:
            self.vector_db = Chroma.from_documents(
                chunks,
                embedding=self.embeddings,
                persist_directory=DB_DIR
            )
            return len(chunks)
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return 0

    def ask(self, question):
        """Get answer with sources"""
        if not self.vector_db:
            return "Please process sources first", []
        
        try:
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": self.qa_prompt},
                return_source_documents=True
            )
            
            result = qa.invoke({"query": question})
            answer = result["result"]
            
            # Get unique sources
            sources = set()
            for doc in result["source_documents"]:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            return answer, list(sources)
        except Exception as e:
            return f"Error answering question: {str(e)}", []

def display_sources(sources, processor):
    """Display sources with guaranteed unique download buttons"""
    if not sources:
        return
    
    with st.expander("📚 Sources"):
        for source in sources:
            if source in processor.source_files:  # PDF file
                file_info = processor.source_files[source]
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<span class="pdf-source">📄 {source}</span>', 
                              unsafe_allow_html=True)
                with col2:
                    with open(file_info["path"], "rb") as f:
                        # Create truly unique key using timestamp + file path hash
                        unique_key = f"dl_{int(time.time()*1000)}_{hash(file_info['path']) & 0xFFFFF}"
                        st.download_button(
                            label="Download",
                            data=f,
                            file_name=source,
                            mime="application/pdf",
                            key=unique_key  # Guaranteed unique
                        )
            
            elif source in processor.source_urls:  # URL
                st.markdown(
                    f'<a href="{source}" class="url-source" target="_blank">🌐 {source}</a>',
                    unsafe_allow_html=True
                )

def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def main():

    
    st.title("📚 SmartScan AI")
    st.caption("Your Intelligent Document Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "url_list" not in st.session_state:
        st.session_state.url_list = []


    with st.sidebar:
        st.header("📂 Source Management")
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Upload PDFs or add web URLs")
        st.markdown("2. Click 'Process Sources'")
        st.markdown("3. Ask questions in the chat")
        st.markdown("---")
        # Tab interface for PDFs and URLs
        tab1, tab2 = st.tabs(["📄 PDF Files", "🌐 Web URLs"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=["pdf"],
                accept_multiple_files=True,
                help=f"Select multiple PDF files (max {MAX_FILE_SIZE_MB}MB each)",
                key="pdf_uploader"
            )
        
        with tab2:
            url_input = st.text_input(
                "Enter URL to analyze",
                placeholder="https://example.com",
                key="url_input"
            )
            url_list = st.session_state.get("url_list", [])
            
            if st.button("Add URL", key="add_url") and url_input:
                if validate_url(url_input):
                    if url_input not in url_list:
                        url_list.append(url_input)
                        st.session_state.url_list = url_list
                        st.success(f"Added URL: {url_input}")
                    else:
                        st.warning("URL already added")
                else:
                    st.error("Please enter a valid URL (including http:// or https://)")
            
            if url_list:
                st.markdown("**URLs to process:**")
                urls_to_remove = []
                
                for i, url in enumerate(url_list):
                    col1, col2 = st.columns([4, 1])
                    col1.markdown(f"- {url}")
                    
                    # Improved delete button with better styling
                    if col2.button("🗑️", 
                                key=f"del_url_{i}",
                                type="secondary"):  # Makes it more prominent
                        urls_to_remove.append(url)
                
                # Remove selected URLs from the list
                url_list = [url for url in url_list if url not in urls_to_remove]
                st.session_state.url_list = url_list  # Update session state
                
                # Show confirmation if URLs were removed
                if urls_to_remove:
                    st.success(f"Removed {len(urls_to_remove)} URL(s)")
                    st.rerun()  # Refresh to show updated list immediately

        
        st.markdown("---")
        
        if st.button("Process Sources", key="process_btn"):
            has_pdfs = uploaded_files and len(uploaded_files) > 0
            has_urls = url_list and len(url_list) > 0
            
            if not has_pdfs and not has_urls:
                st.error("Please add PDFs or URLs to process")
            else:
                with st.spinner("Analyzing content..."):
                    chunk_count = st.session_state.processor.process_sources(
                        pdf_files=uploaded_files if has_pdfs else None,
                        urls=url_list if has_urls else None
                    )

                    if chunk_count > 0:
                        st.session_state.processed = True
                        st.session_state.messages = []
                        st.success("Sources processed successfully!")
                    else:
                        st.error("Failed to process sources")
        
        st.markdown("---")

    # Main chat container
    chat_container = st.container()
    
    # Display chat messages with emoji avatars
    with chat_container:
        for message in st.session_state.messages:
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(
                    f'<div class="{"user-message" if message["role"] == "user" else "assistant-message"} chat-message">'
                    f'{message["content"]}</div>', 
                    unsafe_allow_html=True
                )
                if "sources" in message:
                    display_sources(message["sources"], st.session_state.processor)

    # Chat input with better UX
    if prompt := st.chat_input("Ask about your documents..."):
        if not st.session_state.processed:
            st.warning("⚠️ Please process sources first")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(
                        f'<div class="user-message chat-message">{prompt}</div>', 
                        unsafe_allow_html=True
                    )
            
            # Get and display assistant response
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing content..."):
                        answer, sources = st.session_state.processor.ask(prompt)
                        st.markdown(
                            f'<div class="assistant-message chat-message">{answer}</div>', 
                            unsafe_allow_html=True
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        display_sources(sources, st.session_state.processor)


if __name__ == "__main__":
    main()