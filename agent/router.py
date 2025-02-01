from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["snowflake_store", "doc_store"] = Field(
        ...,
        description="Given a user question choose to route it to snowflake_store or a doc_store.",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a structured set of data in snowflake (snowflake_store) or to a document store that has document search (doc_store).
The snowflake_store contains tables for patients, survey analyts, survey attempts, survey costs, survey responses, and visits.
The doc_store contains documents and chats around the process and logistics of running surveys, including our current rates, agreements, and process for conducting surveys.
Based on the user question, route the question to the most relevant datasource.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

print(question_router.invoke({"question": "Show me the cost trend of our surveys"}))
print(
    question_router.invoke(
        {
            "question": "Using the survey data response and issue behaviors, explore potential ways we could reduce cost."
        }
    )
)
print(
    question_router.invoke(
        {
            "question": "Analyze if there's any response patterns that could help us reduce survey costs"
        }
    )
)
print(
    question_router.invoke(
        {"question": "What is our current rates from our comm platform for surveys?"}
    )
)
