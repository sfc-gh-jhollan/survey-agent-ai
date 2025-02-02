from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


from cortex_search_retriever import CortexSearchRetriever
from router import question_router


# test the retriever
retriever = CortexSearchRetriever(
    documents=[],
    k=5,
)


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"data": documents, "question": question}


### Edges ###


def route_question(state):
    """
    Route question to snowflake_store or doc_store

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "snowflake_store":
        print("---ROUTE QUESTION TO SNOWFLAKE_STORE---")
        return "snowflake_store"
    elif source.datasource == "doc_store":
        print("---ROUTE QUESTION TO RAG---")
        return "doc_store"


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    question = state["question"]
    data = state["data"]

    # RAG generation
    generation = rag_chain.invoke({"context": data, "question": question})
    return {"data": data, "question": question, "generation": generation}
