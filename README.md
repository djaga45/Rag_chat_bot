
## Project Preview

User Uploads PDF/DOCX 
        ↓
Document Loader (PyPDFLoader / Docx2txtLoader)
        ↓
Text Splitter (Chunking)
        ↓
Embeddings (nomic-embed-text)
        ↓
Vector Store (Chroma DB)
        ↓
Retriever (Top-k similarity search)
        ↓
LLM (deepseek-r1:8b) + Prompt
        ↓
Final Answer
