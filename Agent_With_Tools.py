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
groq_api_key = os.getenv("GROQ_API_KEY")
STOCK_API_KEY = os.getenv("STOCK_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


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
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    if not ALPHAVANTAGE_API_KEY:
        return {"error": "ALPHAVANTAGE_API_KEY not set in environment."}
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHAVANTAGE_API_KEY},
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


# def make_generator_tool(file_path: str):
#     """
#     Build a RAG tool bound to a specific uploaded document.
#     The tool name is fixed as 'generator'; only one document is active at a time.
#     """

#     @tool
#     def generator(query: str) -> str:
#         """
#         Answer questions about the uploaded document using retrieval-augmented generation.
#         Use this tool whenever the user asks about the content of the uploaded file.
#         """
#         try:
#             retriever = get_retriever(file_path, k=5)
#             docs = retriever.invoke(query)

#             if not docs:
#                 return "I don't know based on the provided document."

#             context_text = "\n\n".join(doc.page_content for doc in docs)

#             prompt = (
#                 "You are a document-grounded assistant.\n\n"
#                 "Answer ONLY from the provided context.\n"
#                 "If the context does not contain enough information, say exactly:\n"
#                 '"I don\'t know based on the provided document."\n\n'
#                 f"User question:\n{query}\n\n"
#                 f"Retrieved context:\n{context_text}\n\n"
#                 "Answer:"
#             )
#             llm2 = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)
#             return llm2.invoke(prompt).content

#         except Exception as e:  # noqa: BLE001
#             return f"RAG tool error: {e}"

#     return generator


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BASE_SYSTEM_MESSAGE = """
You are a precision-oriented AI assistant.

Available tools:
- date_time: return current system date and time.
- current_weather: get live weather for any city.
- get_stock_price: get a real-time stock quote by ticker symbol.
- internet_search: search the live web for current information.
- generator: answer questions about the uploaded document (document mode only).

Rules:
- Always use date_time before answering time-sensitive questions.
- Use current_weather for any weather-related query.
- Use get_stock_price for stock or market queries.
- Use internet_search for questions requiring up-to-date web information.
- Use generator ONLY when a document has been uploaded and the user's question is about it.
- If a tool returns an error or empty result, report this clearly.
- Be concise, accurate, and helpful.
""".strip()

DOCUMENT_ADDENDUM = """

Document mode is active:
- For any question about the uploaded file, always prefer the generator tool.
- Do not answer from general knowledge when the user asks about the document.
""".strip()


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def make_generator_tool(file_path: str, groq_api_key: str | None = None):
    @tool
    def generator(query: str) -> str:
        """Answer questions about the uploaded document using retrieval-augmented generation."""
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
                f"Retrieved context:\n{context_text}\n\nAnswer:"
            )
            llm2 = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)
            return llm2.invoke(prompt).content
        except Exception as e:
            return f"RAG tool error: {e}"
    return generator


def build_workflow(file_path: str | None = None, groq_api_key: str | None = None) -> object:
    tools: list = [current_weather, internet_search, get_stock_price, date_time]
    system_message = BASE_SYSTEM_MESSAGE
    if file_path:
        tools.append(make_generator_tool(file_path, groq_api_key=groq_api_key))
        system_message = f"{BASE_SYSTEM_MESSAGE}\n\n{DOCUMENT_ADDENDUM}"

    llm1 = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)
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
