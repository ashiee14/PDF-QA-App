# 📄 Llama-3.3-70B Document RAG QA

### ⚡ High-Performance RAG-based PDF Question Answering using Groq + LangChain + Streamlit

An advanced **Retrieval Augmented Generation (RAG)** based PDF Question
Answering application powered by:

-   🧠 **Llama-3.3-70B-Versatile (via Groq API)**
-   🔗 **LangChain**
-   📚 **Chroma Vector Database**
-   🤗 **HuggingFace Embeddings**
-   🌐 **Streamlit UI**

This app allows users to upload PDF documents and ask questions based on
their content using a blazing-fast LLM inference engine powered by
**Groq**.

------------------------------------------------------------------------

# 🚀 Features

-   📤 Upload one or multiple PDF files
-   🔍 Automatic document chunking & embedding
-   🗄️ Vector storage using ChromaDB
-   ⚡ Lightning-fast inference with Groq
-   🤖 Llama-3.3-70B model integration
-   💬 Conversational question answering
-   🧠 Context-aware RAG pipeline
-   🌍 Streamlit Web Interface
-   ☁️ Deployable on Streamlit Cloud

------------------------------------------------------------------------

# 🏗️ Architecture

User Question\
↓\
Retriever (Chroma)\
↓\
Relevant Document Chunks\
↓\
Llama-3.3-70B (Groq API)\
↓\
Final Answer

------------------------------------------------------------------------

# 📦 Tech Stack

  Component      Technology
  -------------- --------------------------------
  LLM            Llama-3.3-70B-Versatile (Groq)
  Framework      LangChain
  Embeddings     HuggingFace Transformers
  Vector Store   ChromaDB
  Backend        Python
  UI             Streamlit
  Deployment     Streamlit Cloud

------------------------------------------------------------------------

# 🛠️ Installation (Local Setup)

## 1️⃣ Clone Repository

``` bash
git clone https://github.com/your-username/pdfQAapp.git
cd pdfQAapp
```

## 2️⃣ Create Virtual Environment (Python 3.10 Recommended)

``` bash
py -3.10 -m venv venv
venv\Scripts\activate
```

## 3️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔑 Environment Variables

Create a `.env` file in root directory:

    GROQ_API_KEY=your_groq_api_key_here

------------------------------------------------------------------------

# 🌐 Run Application

``` bash
streamlit run app.py
```

Then open:\
http://localhost:8501

------------------------------------------------------------------------

# ☁️ Deploy on Streamlit Cloud

1.  Push code to GitHub\
2.  Open Streamlit Cloud\
3.  Deploy new app\
4.  Add secret in **App Settings → Secrets**

```{=html}
<!-- -->
```
    GROQ_API_KEY = "your_actual_key_here"

5.  Deploy 🚀

------------------------------------------------------------------------

# 🧠 Model Details

### 🔥 Llama-3.3-70B-Versatile

-   70 Billion parameters
-   Optimized for reasoning & instruction following
-   High-speed inference via Groq
-   Low latency RAG response generation

------------------------------------------------------------------------

# 📁 Project Structure

    pdfQAapp/
    │
    ├── app.py
    ├── requirements.txt
    ├── .env
    ├── README.md
    └── vectorstore/

------------------------------------------------------------------------

# ⚠️ Important Notes

-   Use **Python 3.10 (64-bit)** for best compatibility
-   Do NOT use 32-bit Python
-   Keep your API keys secure
-   Never upload `.env` to GitHub
-   Use `.gitignore` to protect secrets

------------------------------------------------------------------------

# 👩‍💻 Author

**Aishwarya Gupta**\
B.Tech CSE (AI Specialization)\
RAG \| GenAI \| LangChain \| LLM Applications

------------------------------------------------------------------------

# ⭐ If You Like This Project

Give it a ⭐ on GitHub\
Fork & Improve\
Use it in your AI portfolio

------------------------------------------------------------------------

# 🏁 Conclusion

This project demonstrates a production-ready:

✅ Retrieval Augmented Generation system\
✅ Groq LLM integration\
✅ LangChain orchestration\
✅ Real-time Streamlit deployment

A powerful addition to any AI/ML engineer's portfolio.
