from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from cortex_search_retriever import CortexSearchRetriever

# test the retriever
retriever = CortexSearchRetriever(
    documents=[],
    k=5,
)
docs = retriever.invoke("What are the rates we pay for different survey options?")
# print(docs)
