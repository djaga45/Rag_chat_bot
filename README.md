
| Component          | Technology                     |
|--------------------|--------------------------------|
| Framework          | LangChain + Streamlit          |
| LLM                | deepseek local Model
| Embeddings         | Dense retrival                 |
| Vector Database    | ChromaDB                       |
| Text Splitting     | RecursiveCharacterTextSplitter |
| Document Loading   | PyPDFLoader + Docx2txtLoader   |



## 2. Create virtual environment & install dependencies BasH

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

## 3. Install Ollama models
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text

## 4. Run the application
streamlit run app.py


## Project Preview

1)User Uploads PDF/DOCX 

2)Document Loader (PyPDFLoader / Docx2txtLoader)
        
3)Text Splitter (Chunking)
        
4)Embeddings (nomic-embed-text)
        
5)Vector Store (Chroma DB)
        
6)Retriever (Top-k similarity search)
        
7)LLM (deepseek-r1:8b) + Prompt
        
8)Final Answer



