from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document, Blob
from langgraph.types import StreamWriter


from pydantic import BaseModel, Field

from cortex_search_retriever import CortexSearchRetriever
from cortex_analyst_retriever import call_cortex_analyst
from router import question_router

# test the retriever
retriever = CortexSearchRetriever(
    documents=[],
    k=5,
)


def retrieve(state, writer: StreamWriter):
    """
    Retrieve documents

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    writer("Looking at documents...")
    question = state.question

    # Retrieval
    documents = retriever.invoke(question)
    return {"data": documents, "question": question}


### Edges ###


def route_question(state, writer: StreamWriter):
    """
    Route question to snowflake_store or doc_store

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Next node to call
    """
    writer("Evaluating how to answer question...")
    print("---ROUTE QUESTION---")
    q: str = state.question
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


def generate(state, writer: StreamWriter):
    """
    Generate answer

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    writer("Generating response...")

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
    question = state.question
    data = state.data
    # print("DATA ### ", data)

    # RAG generation
    generation = rag_chain.invoke({"data": data, "question": question})
    return {"data": data, "question": question, "generation": generation}


def generate_analysis_prompts(state, writer: StreamWriter):
    """
    Based on the question, generate a number of prompts to explore and help answer the analysis

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): New key added to state, analysis_prompts, that contains an array of generated prompts
    """
    print("---GENERATE ANALYSIS PROMPTS---")
    writer("Exploring data to analyze...")

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


        Questions: Analyze how I can save costs by being more efficient with survey methods with patient surveys
        Returned 5 Prompts:
        - What is the cost of survey response by survey method?
        - What is the survey response rate by age based on survey method?
        - What is the survey response rate by gender based on survey method?
        - What is the survey response rate by location based on survey method?
        - What is the overall likelyhood of response based on survey method?
                     
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
    question = state.question
    if hasattr(state, "data"):
        data = state.data
    else:
        data = []

    response = chain.invoke(
        {"context": data, "question": question, "semantic_model": semantic_model}
    )
    # print(response)
    analysis_prompts = response.prompts
    return {"data": data, "question": question, "analysis_prompts": analysis_prompts}


def exec_sql_analysis(state, writer: StreamWriter):
    print("---EXEC ANALYSIS PROMPTS---")
    writer("Executing SQL queries to analyze data...")
    if hasattr(state, "data"):
        d = state.data
    else:
        d = []
    if hasattr(state, "analysis_prompts") and len(state.analysis_prompts) > 0:
        analysis_prompts = state.analysis_prompts
        prompt_lenght = len(analysis_prompts)
        counter = 0
        for prompt in analysis_prompts:
            counter = counter + 1
            writer(f"Executing query {counter} of {prompt_lenght}...")
            res = call_cortex_analyst(prompt)
            d.append(Document(page_content=res))
    else:
        raise Exception("No analysis prompts to execute")
    return {
        "data": d,
        "question": state.question,
        "analysis_prompts": [],
    }


def analyze_results(state, writer: StreamWriter):
    print("---ANALYZE RESULTS OF ANALYSIS PROMPTS---")
    writer("Analyzing results of data queries...")
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
    question = state.question
    if hasattr(state, "data"):
        data = state.data
    else:
        data = []

    # RAG generation
    response = chain.invoke({"data": data, "question": question})
    # print(response)
    data.append(Document(page_content=response.content))
    return {
        "data": data,
        "question": question,
    }


def decide_to_reanalyze(state, writer: StreamWriter):
    """
    Decide if we should analyze the results or not

    Args:
        state (GraphState): The current graph state

    Returns:
        str: Next node to call
    """
    writer("Deciding what to do next...")
    if hasattr(state, "prompts_to_review") and len(state.prompts_to_review) > 0:
        print("---DECIDE TO REVISE---")
        return "revise"
    else:
        print("---DECIDE TO COMPLETE---")
        return "complete"


def revise_analysis_prompts(state, writer: StreamWriter):
    print("---ANALYZE RESULTS OF ANALYSIS PROMPTS---")
    writer("Exploring a few more angles...")
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
    question = state.question
    if hasattr(state, "data"):
        data = state.data
    else:
        data = []

    # RAG generation
    response = chain.invoke({"data": data, "question": question})
    # print(response)
    data.append(Document(page_content=response.content))
    return {
        "data": data,
        "question": question,
    }
