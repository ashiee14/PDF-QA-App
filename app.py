import os
import streamlit as st

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in Streamlit Secrets.")
    st.stop()

from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Document RAG QA", layout="centered")
st.title("Llama-3.3-70B Document RAG QA with Groq and LangChain")

# Session state flag
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None and not st.session_state.doc_processed:

    save_path = os.path.join(working_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document... This may take a few seconds..."):
        process_document_to_chroma_db(uploaded_file.name)

    st.session_state.doc_processed = True
    st.success("Document processed successfully!")

# Only show question box AFTER processing
if st.session_state.doc_processed:

    user_question = st.text_input("Enter your question")

    if st.button("Get Answer"):

        with st.spinner("Generating answer..."):
            answer = answer_question(user_question)

        st.markdown("### Llama-3.3-70B Answer:")
        st.write(answer)

