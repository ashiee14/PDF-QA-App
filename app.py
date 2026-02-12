import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="PDF QA with Llama 3.3", layout="wide")

st.title("📄 Llama-3.3-70B Document RAG QA")

# ✅ Use Streamlit secrets (NOT os.getenv)
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()

llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0
)

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # ✅ Use temporary file (better for cloud)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    # ✅ Use Chroma instead of FAISS (Cloud-friendly)
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            response = qa.invoke({"query": query})
            st.success(response["result"])
