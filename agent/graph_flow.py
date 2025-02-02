from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


from pydantic import BaseModel, Field

from cortex_search_retriever import CortexSearchRetriever
from cortex_analyst_retriever import call_cortex_analyst
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
    q = state["question"]
    source = question_router.invoke({"question": q})
    if source.action == "snowflake_store":
        print("---ROUTE QUESTION TO SNOWFLAKE_STORE---")
        return "snowflake_store"
    elif source.action == "doc_store":
        print("---ROUTE QUESTION TO RAG---")
        return "doc_store"
    elif source.action == "data_analysis":
        print("---ROUTE QUESTION TO GENERATE ANALYSIS PROMPTS---")
        return "data_analysis"


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")

    prompt = ChatPromptTemplate.from_template(
        """
        Given the following question: {question}
        And the included data below, generate an answer to the question.
        If there isn't enough context or detail, respond that you do not know or do not have enough data.

                               
        ### DATA ###
        Data to assist you in answering the question:
        {data}
        """,
    )

    # LLM
    llm = ChatOpenAI(model_name="o3-mini")

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    question = state["question"]
    data = state["data"]
    print("DATA ### ", data)

    # RAG generation
    generation = rag_chain.invoke({"data": data, "question": question})
    return {"data": data, "question": question, "generation": generation}


def generate_analysis_prompts(state):
    """
    Based on the question, generate a number of prompts to explore and help answer the analysis

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, analysis_prompts, that contains an array of generated prompts
    """
    print("---GENERATE ANALYSIS PROMPTS---")

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        Given the following question: {question}
        Generate a series of prompts that can be used to explore the data that could surface
        info on how to answer the question. The intent is that we query the data on a few different
        dimensions to see if there are any trends or insights that could help answer the question.
        Ideally prompts will drive towards queries that could show patterns or correlation.

        Consider many potential prompts, but only return at most 5 that will be explored.

        ### 
        EXAMPLES:

        Question: Explore patient satisfaction trends over the last 6 months.
        Considered Prompts:
        - What is the average, min, and max patient satisfaction score based on visit reason?
        - What is the average, min, and max patient satisfaction score based on visit outcome?
        - What is the average, min, and max patient satisfaction score based on visit cost?
        - What is the average, min, and max patient satisfaction score grouped by months?
        - What is the average, min, and max patient satisfaction score grouped by patient age?
        - What is the average, min, and max patient satisfaction score grouped by patient gender?
        - What is the average, min, and max patient satisfaction score grouped by patient location?
        Returned 5 Prompts:
        - What is the average, min, and max patient satisfaction score grouped by patient gender?
        - What is the average, min, and max patient satisfaction score based on visit reason?
        - What is the average, min, and max patient satisfaction score based on visit outcome?
        - What is the average, min, and max patient satisfaction score based on visit cost?
        - What is the average, min, and max patient satisfaction score grouped by patient age?

                               
        ###
        Context from the current conversation or other additions from agents:
        {context}

        ### 
        Semantic model for the data that has the different tables and columns that can be queried:
        {semantic_model}
        """,
    )

    with open("data/survey_model.yaml") as f:
        semantic_model = f.read()

    # LLM
    llm = ChatOpenAI(model_name="o3-mini")

    class PromptsArray(BaseModel):
        """A list of prompts generated to help drive analysis."""

        prompts: list[str] = Field(
            ...,
            description="An array of prompts generated based on the user question and context.",
        )

    structured_llm_generator = llm.with_structured_output(PromptsArray)

    # Chain
    chain = prompt | structured_llm_generator
    question = state["question"]
    if "data" in state:
        data = state["data"]
    else:
        data = "None"

    response = chain.invoke(
        {"context": data, "question": question, "semantic_model": semantic_model}
    )
    print(response)
    analysis_prompts = response.prompts
    return {"data": data, "question": question, "analysis_prompts": analysis_prompts}


def exec_sql_analysis(state):
    print("---EXEC ANALYSIS PROMPTS---")
    if "analysis_prompts" in state and len(state["analysis_prompts"]) > 0:
        analysis_prompts = state["analysis_prompts"]
        results = []
        for prompt in analysis_prompts:
            res = call_cortex_analyst(prompt)
            results.append(res)
    else:
        raise Exception("No analysis prompts to execute")
    results_str = "### Data Results \n".join(results)
    return {
        "data": Document(page_content=results_str),
        "question": state["question"],
        "analysis_prompts": [],
    }


def analyze_results(state):
    print("---ANALYZE RESULTS OF ANALYSIS PROMPTS---")
    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        Given the following question: {question}
        We have run a series of queries to try to answer this question or shed light on it.
        Based on all of the results we have in the data below, generate a summary of insights.

        Be very clear and verbose.

        ### Data
        {data}
        """,
    )

    # LLM
    llm = ChatOpenAI(model_name="o3-mini")

    # Chain
    chain = prompt | llm
    question = state["question"]
    if "data" in state:
        data = state["data"]
    else:
        data = "None"

    # RAG generation
    response = chain.invoke({"data": data, "question": question})
    print(response)
    return {"data": Document(page_content=response.content), "question": question}
