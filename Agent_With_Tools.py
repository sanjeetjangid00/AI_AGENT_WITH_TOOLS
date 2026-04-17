from __future__ import annotations
import os
import time
from functools import lru_cache
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STOCK_API_KEY = os.getenv("STOCK_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# --- Models ---
llm1 = ChatGroq(model="openai/gpt-oss-120b")
llm2 = ChatGroq(model="openai/gpt-oss-20b")

# --- Embeddings ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Document loading & retrieval
# ---------------------------------------------------------------------------

def file_loader(file_path: str) -> list:
    file_path = str(file_path)
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        loader = Docx2txtLoader(file_path)
    return loader.load()


def text_splitter(file_path: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = file_loader(file_path)
    return splitter.split_documents(docs)


@lru_cache(maxsize=8)
def create_vector_db(file_path: str) -> FAISS:
    chunks = text_splitter(file_path)
    return FAISS.from_documents(chunks, embedding)


@lru_cache(maxsize=8)
def get_retriever(file_path: str, k: int = 5):
    """Return a cached FAISS retriever for the given file."""
    db = create_vector_db(file_path)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def current_weather(city: str) -> dict | str:
    """Get current weather for a city using Open-Meteo (no API key required)."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=15,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"City '{city}' not found."

        result = geo_data["results"][0]
        lat, lon = result["latitude"], result["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=15,
        )
        weather_response.raise_for_status()
        return weather_response.json()

    except requests.RequestException as e:
        return f"Weather tool error: {e}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock quote for a ticker symbol using Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY not set in environment."}
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Stock tool error: {e}"}


@tool
def date_time() -> dict:
    """Return the current system date and time."""
    return {"current_date_time": time.ctime()}


internet_search = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en", max_results=5)
)


def make_generator_tool(file_path: str):
    """
    Build a RAG tool bound to a specific uploaded document.
    The tool name is fixed as 'generator'; only one document is active at a time.
    """

    @tool
    def generator(query: str) -> str:
        """
        Answer questions about the uploaded document using retrieval-augmented generation.
        Use this tool whenever the user asks about the content of the uploaded file.
        """
        try:
            retriever = get_retriever(file_path, k=5)
            docs = retriever.invoke(query)

            if not docs:
                return "I don't know based on the provided document."

            context_text = "\n\n".join(doc.page_content for doc in docs)

            prompt = (
                "You are a document-grounded assistant.\n\n"
                "Answer ONLY from the provided context.\n"
                "If the context does not contain enough information, say exactly:\n"
                '"I don\'t know based on the provided document."\n\n'
                f"User question:\n{query}\n\n"
                f"Retrieved context:\n{context_text}\n\n"
                "Answer:"
            )
            return llm2.invoke(prompt).content

        except Exception as e:  # noqa: BLE001
            return f"RAG tool error: {e}"

    return generator


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# System prompt  (improved)
# ---------------------------------------------------------------------------

BASE_SYSTEM_MESSAGE = """
You are a precision-oriented AI assistant with access to the following tools:

TOOLS:
- date_time         → Returns the current system date and time.
- current_weather   → Returns live weather data for a given city or location.
- get_stock_price   → Returns a real-time stock quote for a given ticker symbol.
- internet_search   → Searches the live web for current or general information.
- generator         → Queries the content of the user's uploaded document using
                      retrieval. Use ONLY when a document has been uploaded AND
                      the user's question is specifically about that document.

TOOL SELECTION RULES (apply in order):
1. If the question is about the uploaded document → use `generator` first.
2. If the question is about weather → use `current_weather` only.
3. If the question is about a stock, ticker, or market price → use `get_stock_price` only.
4. If the question requires the current date/time → use `date_time` first.
5. For all other time-sensitive or factual questions → use `internet_search`.
6. If no tool is clearly needed, answer from your own knowledge directly.

MULTI-INTENT QUERIES:
- If the user asks multiple questions in one message, handle each sub-question
  with the appropriate tool independently.
- Call each tool only once per sub-question unless the user provides new input.

TOOL USAGE DISCIPLINE:
- Never call a tool unless it is clearly required by the query.
- Never call the same tool more than once for the same question.
- Never loop between tools trying to construct an answer.
- If a tool returns an error, empty result, or irrelevant data, stop retrying
  and respond with: "I wasn't able to retrieve that information right now.
  Here's what I know: [fallback answer or honest acknowledgment]."
- Never guess or fabricate live or real-time data (prices, weather, time, etc.).

RESPONSE STYLE:
- Be concise, accurate, and direct.
- Use bullet points for lists; prose for explanations.
- Do not over-explain tool selection or internal reasoning to the user.
- If the exact answer cannot be found, say so clearly and briefly.
""".strip()

DOCUMENT_ADDENDUM = """

DOCUMENT MODE (active — a file has been uploaded):
- Always use the `generator` tool for any question about the uploaded document.
- Do not answer document-related questions from general knowledge.
- If `generator` returns no useful result, respond with:
  "I couldn't find that in the uploaded document."
""".strip()


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_workflow(file_path: str | None = None) -> object:
    """
    Compile a LangGraph workflow.

    When file_path is provided, a RAG tool for that document is injected and
    the system prompt is extended with document-mode instructions.
    """
    tools: list = [current_weather, internet_search, get_stock_price, date_time]

    system_message = BASE_SYSTEM_MESSAGE
    if file_path:
        tools.append(make_generator_tool(file_path))
        system_message = f"{BASE_SYSTEM_MESSAGE}\n\n{DOCUMENT_ADDENDUM}"

    llm_with_tools = llm1.bind_tools(tools)

    def chat_node(state: ChatState) -> dict:
        messages = [SystemMessage(content=system_message)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=InMemorySaver())
