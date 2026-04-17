# 🚀 LangGraph Multi-Tool RAG Assistant

A **production-style AI assistant** built using **LangGraph + LangChain**, capable of:

- 📄 Document-based Q&A (RAG)
- 🌦️ Real-time weather data
- 📈 Stock price retrieval
- 🌐 Internet search
- ⏱️ Current date & time

This system intelligently selects tools based on user queries and ensures **accurate, non-hallucinated responses**.

---

## 🧠 Key Features

- **RAG (Retrieval-Augmented Generation)**
  - Supports PDF, TXT, and DOCX
  - Uses FAISS vector database
  - HuggingFace embeddings

- **Multi-tool Agent**
  - Weather API (Open-Meteo)
  - Stock API (Alpha Vantage)
  - DuckDuckGo Search
  - System date/time

- **LangGraph Workflow**
  - Stateful conversation
  - Tool routing with conditions
  - Memory checkpointing

- **Smart Tool Selection**
  - Uses strict rules to avoid unnecessary tool calls
  - Handles multi-intent queries efficiently

---

## 🏗️ Tech Stack

- **LLM**: Groq (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`)
- **Frameworks**: LangChain, LangGraph
- **Vector DB**: FAISS
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **APIs**:
  - Open-Meteo (Weather)
  - Alpha Vantage (Stocks)
  - DuckDuckGo (Search)

---

## 📂 Project Structure
```
├── main.py / app.py
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/sanjeetjangid00/AI_AGENT_WITH_TOOLS.git
cd AI_AGENT_WITH_TOOLS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create .env file
```
GROQ_API_KEY=your_groq_key
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key
HF_TOKEN=your_huggingface_token
```
▶️ Usage
Run the assistant
```bash
workflow = build_workflow(file_path="your_document.pdf")  # for RAG
# OR
workflow = build_workflow()  # without document
```
Then pass messages into the graph.

### 📄 Supported Inputs
Type	Supported
- PDF	✅
- TXT	✅
- DOCX	✅

### 🔍 How It Works
#### 1. Document Processing
Load → Split → Embed → Store in FAISS

#### 2. Query Handling
LangGraph routes query
Chooses:
- Tool OR
- Direct LLM response
  
#### 3. RAG Flow
- Retrieve top-k chunks
- Inject into prompt
- Generate grounded answer
  
🛠️ Tools Available
### Tool	Description
```bash
current_weather	Fetch live weather
get_stock_price	Get stock data
date_time	System time
internet_search	Web search
generator	Document Q&A (RAG)
```

### 🧩 Workflow Architecture
```bash
User Query
    ↓
Chat Node (LLM + Tool Binding)
    ↓
Tool Decision (Conditional Edge)
    ↓
Tool Execution (if needed)
    ↓
Response Generation
```

## ⚠️ Important Design Decisions
- ❌ No hallucination of real-time data
- ❌ No repeated tool calls
- ❌ No fallback to web for document queries
- ✅ Strict tool usage rules
- ✅ Cached retriever (performance optimized)

**👨‍💻 Author**
Sanjeet Jangid
