# 🤖 Customer Support GenAI Agent

An AI-powered assistant built to handle queries in **Banking**, **Insurance**, and **Finance** domains using **LLM** and **text embeddings**. This project leverages cutting-edge technologies like **Groq streaming**, **LangChain**, and **FAISS vector search** to offer smart, instant, and contextual customer support.

---

## 📌 Problem Statement

In today's digital era, users expect real-time and accurate support. Manual handling of repeated queries leads to:

- Delays in responses
- Increased operational costs
- Poor customer experience

---

## ✅ Solution

This GenAI agent uses advanced **language models** and **document embeddings** to understand user intent and deliver accurate responses by retrieving relevant content from policy documents, FAQs, and guidelines in real time.

---
![Screenshot 2025-06-28 161840](https://github.com/user-attachments/assets/093a62f6-0c15-4c7d-94b0-885220f264ed)

## DEMO 
Open the link for Demo:  https://drive.google.com/file/d/1sTxQ-_5t9sSa1qP-yI9WNiuLtog_6Duz/view?usp=drive_link


## 🧠 How It Works (Flow)

1. **User Input:** User enters a query via chatbot/web UI  
2. **Embedding Generation:** Query is converted into vector using a Text Embedding Model  
3. **Similarity Search:** FAISS vector store fetches the most relevant document chunks  
4. **Context Injection:** Retrieved chunks are passed to the LLM (via LangChain)  
5. **Response Generation:** LLM generates a final contextual answer  
6. **Output:** Answer is displayed to the user

---

## 🚀 Features

- 📎 Document-based question answering (banking, insurance, finance)
- ⚡ Fast responses using Groq LPU (streaming enabled)
- 🔍 FAISS for semantic search
- 🤖 LangChain for chaining logic
- 🧩 Modular code (easy to extend or fine-tune)

---

## 🛠 Tech Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| UI               | Streamlit                         |
| Backend Logic    | LangChain, Python                 |
| Embedding Model  | `text-embedding-3-small` (OpenAI) |
| Vector DB        | FAISS                             |
| LLM Provider     | Groq (LLM: Mixtral / LLaMA3)      |
| Deployment       | Local / GitHub / Streamlit Cloud  |

---

## 📂 Project Structure

```
genai/
│
├── data/              # PDF docs
├── app.py             # Main Streamlit app
├── ingest.py          # Loads and vectorizes documents
├── .env               # API keys (never push this!)
├── utils.py           # Helper functions
├── requirements.txt   # Dependencies
└── README.md          # You’re reading it!
```

---

## 🚦 Run Locally

```bash
git clone <repo-url>
cd customer-support-genai-agent
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt
python ingest.py          # Index the documents
streamlit run app.py
```

