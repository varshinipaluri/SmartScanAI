# 📚 SmartScan AI - Smart Document Assistant

## 🚀 Overview
**SmartScan AI** is a smart document assistant that empowers users to upload PDF files or input web URLs. It extracts, processes, and analyzes content, enabling users to ask intelligent questions and receive insightful answers using Google Gemini AI.

## ✨ Features
- 📄 Upload and analyze PDF documents  
- 🌐 Extract and process content from web URLs  
- 🧠 AI embeddings using Google Gemini  
- 💬 Conversational Q&A interface  
- 🗂️ Contextual source tracking and preview  
- 💾 Downloadable sources and vector storage via ChromaDB  
- 🎨 Streamlit-powered modern UI  

## 🧰 Tech Stack
| Layer        | Technology                        |
|--------------|------------------------------------|
| Frontend     | Streamlit                          |
| Backend      | Python                             |
| LLM & Embedding | Google Generative AI (`gemini-1.5-flash`, `embedding-001`) |
| Vector DB    | ChromaDB                           |
| Parsing      | PyPDF2, BeautifulSoup              |
| NLP Toolkit  | LangChain                          |

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/varshinipaluri/SmartScanAI.git
   cd SmartScanAI
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Your Environment Variables**
   Create a `.env` file in the root directory and add your [Google Gemini API Key](https://aistudio.google.com/app/apikey):
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 🧪 Usage

1. **Upload PDFs** or paste **web URLs** from the sidebar.  
2. Click **“Process Sources”** to extract and vectorize content.  
3. Ask questions in the chat below. The assistant provides answers with clickable source references and downloads.

## 📂 File Structure
```
📁 SmartScanAI
├── final_llm.py                 # Main Streamlit app
├── requirements.txt       # Python packages
├── .env                   # Environment variables (API Key)
├── uploaded_files/        # Directory for uploaded PDFs
├── chroma_db/             # Chroma vector database files
├── README.md              # Project documentation
```

## ⚠️ Troubleshooting

- ❌ **Not responding to questions?** Make sure you've clicked "Process Sources" after uploading.  
- ❗ **GEMINI_API_KEY error?** Check your `.env` file and key validity.  
- 🧾 **PDF not processed?** Ensure it’s not encrypted or image-scanned.  

## 🧠 Future Plans 
- 🧾 Support for DOCX, TXT, and more file types  
- 🧵 Persistent session memory with user history  
- ☁️ Cloud storage integration (Google Drive, Dropbox)  
- 🔍 Document structure preview and smart highlights  
.
**SmartScan AI** — Transforming how you interact with documents using the power of AI.
