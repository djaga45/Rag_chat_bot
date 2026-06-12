import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import tempfile

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")
st.markdown("**Upload your PDF or DOCX file and start chatting**")

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    with st.spinner("Processing document..."):
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = Docx2txtLoader(temp_path)

            docs = loader.load()

            if not docs or not docs[0].page_content.strip():
                st.error(" Could not extract text from the document.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = text_splitter.split_documents(docs)

                # Use proper embedding model
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

                st.success(f"Document processed successfully! ({len(chunks)} chunks)")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask any question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

                llm = ChatOllama(model="deepseek-r1:8b", temperature=0.3)

                template = """Answer the question based only on the following context.
If you don't know the answer, just say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

                prompt_template = ChatPromptTemplate.from_template(template)

                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                response = chain.invoke(prompt)
                st.markdown(response)
            except Exception as e:
                st.error("Please upload a document first.")

    st.session_state.messages.append({"role": "assistant", "content": response})
