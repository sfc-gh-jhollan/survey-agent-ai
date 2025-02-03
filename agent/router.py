from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


# Data model
class RouteQuery(BaseModel):
    """Route a user query to route to the right next action."""

    action: Literal["snowflake_store", "doc_store", "data_analysis"] = Field(
        ...,
        description="Given a user question, choose the next action: lookup data in doc_store, lookup data in snowflake_sore, or run data_analysis.",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# read the semantic model from data/survey_model.yaml
with open("data/survey_model.yaml") as f:
    survey_model = f.read()
# Prompt
system = f"""You are an expert at routing a user question to the right next action. You have four options:
1. snowflake_store: if the question can be answered by querying structured data in Snowflake, this is the right next action. Usually a question about trends or survey data that DOES NOT have the word "analyze", "anaylse", or "explore".
2. doc_store: if the question could be answered by looking into documents indexed in the doc_store, this is the right next action.
3. data_analysis: if the question requires exploring the data, running multiple queries, and doing larger exploration, data_analysis is the next action. Often includes the words "analyse", "analyze", or "explore" in question.
Based on the user question, route the question to the most relevant next action.

### Details
- snowflake_store: this is the semantic model for data that can be answered with snowflake_store
{survey_model}

- doc_store: these are the documents indexed in the doc_store
Survey_Rates
Survey_Agreements
Survey_Process
Compliance_Guidelines
Survey_Optimization_Strategies

### Examples
human: What is the process for survey data collection?
action: doc_store

human: How many surveys have been issued in the last month?
action: snowflake_store

human: Analyze the survey data to find potential cost savings.
action: data_analysis

human: Show me the cost trend of our surveys
action: snowflake_store

human: What's the current trend for survey responses over the last 90 days?
action: snowflake_store
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
