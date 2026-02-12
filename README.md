# 📄 Llama-3.3-70B Document RAG QA

### ⚡ Fast PDF Question Answering using Groq + LangChain + Streamlit

A production-ready **Retrieval Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its content using:

* 🧠 **Llama-3.3-70B-Versatile (Groq API)**
* 🔗 **LangChain**
* 🗄 **Chroma Vector Store**
* 🤗 **HuggingFace Embeddings**
* 🌐 **Streamlit UI**

This app is fully deployable on **Streamlit Cloud**.

---

# 🚀 Features

* 📤 Upload a PDF document
* 🔍 Automatic text chunking
* 🧠 Embedding generation using HuggingFace
* 🗄 In-memory Chroma vector database
* ⚡ High-speed LLM inference via Groq
* 💬 Context-aware Question Answering
* ☁️ Cloud deployable
* 🔐 Secure API key using Streamlit Secrets

---

# 🏗️ Architecture

User Uploads PDF     
   ↓

PyPDFLoader
    
  ↓

Text Splitting
  
  ↓

HuggingFace Embeddings
  
  ↓

Chroma Vector Store
  
  ↓

Retriever
  
  ↓

Llama-3.3-70B (Groq)
  
  ↓

Final Answer

---

# 📦 Tech Stack

| Component    | Technology                          |
| ------------ | ----------------------------------- |
| LLM          | Llama-3.3-70B-Versatile (Groq)      |
| Framework    | LangChain                           |
| Embeddings   | HuggingFace (sentence-transformers) |
| Vector Store | Chroma                              |
| UI           | Streamlit                           |
| PDF Loader   | PyPDF                               |
| Deployment   | Streamlit Cloud                     |

---

# 🛠️ Local Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/pdf-qa-app.git
cd pdf-qa-app
```

## 2️⃣ Create Virtual Environment (Python 3.10 Recommended)

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 📄 requirements.txt

```txt
streamlit==1.33.0
langchain==0.1.20
langchain-core==0.1.53
langchain-community==0.0.38
langchain-text-splitters==0.0.2
langchain-huggingface==0.0.3
langchain-groq==0.1.3
chromadb==0.5.23
sentence-transformers==2.7.0
pypdf==4.2.0
```

---

# 🔑 API Key Setup

This app uses **Streamlit Secrets** instead of `.env`.

## Local Development

Create a file:

```
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

---

# ☁️ Deploy on Streamlit Cloud

1. Push project to GitHub
2. Go to Streamlit Cloud
3. Deploy new app
4. Go to **App Settings → Secrets**
5. Add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

6. Deploy 🚀

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

# 📁 Project Structure

```
pdf-qa-app/
│
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│    └── secrets.toml
└── temp.pdf (generated automatically)
```

---

# ⚠️ Important Notes

* ✅ Use **Python 3.10 (64-bit)**
* ❌ Do NOT use 32-bit Python
* 🔐 Never commit secrets.toml
* 📌 HuggingFace model loads once (cached)
* ☁️ Fully Streamlit Cloud compatible

---

# 🧠 Model Info

### Llama-3.3-70B-Versatile

* 70B parameters
* Strong reasoning capabilities
* Low latency via Groq hardware acceleration
* Excellent for RAG pipelines

---

# 👩‍💻 Author

**Aishwarya Gupta**
B.Tech CSE (AI Specialization)
GenAI | RAG | LangChain | LLM Applications

---

# ⭐ This project demonstrates:

✅ Production RAG pipeline
✅ Secure API management
✅ Groq LLM integration
✅ Vector database usage
✅ Streamlit cloud deployment
