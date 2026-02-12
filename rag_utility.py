import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

#load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

#load the embedding model
embedding = HuggingFaceEmbeddings()

#load the llama model form groq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def process_document_to_chroma_db(file_name):
    #load pdf document using unstructured file loader    
    loader = UnstructuredFileLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    #split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

def answer_question(user_question):
    #load the persistent chroma vector db
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    #create retriever for document search
    retriever = vectordb.as_retriever()
    
    #create retrieval qa chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    
    #get the answer from the chain
    response = qa_chain.invoke({"query":user_question})
    answer= response['result']
    return answer