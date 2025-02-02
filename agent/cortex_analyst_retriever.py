from typing import List

from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.types import StreamWriter


import os
from snowflake.core import Root
from snowflake.snowpark import Session
import pandas as pd

import requests

session = Session.builder.config("connection_name", "pm").getOrCreate()
root = Root(session)

rest_token = session.connection.rest.token
connection = session.connection

DATABASE = "JEFFHOLLAN_DEMO"
SCHEMA = "SURVEY"
STAGE = "MODELS"
FILE = "survey_model.yaml"
HOST = "pm.snowflakecomputing.com"


def call_cortex_analyst(question: str) -> str:

    # Cortex Analyst lookup
    """Calls the REST API and returns the response."""
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
        "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
    }
    resp = requests.post(
        url=f"https://{HOST}/api/v2/cortex/analyst/message",
        json=request_body,
        headers={
            "Authorization": f'Snowflake Token="{rest_token}"',
            "Content-Type": "application/json",
        },
    )
    request_id = resp.headers.get("X-Snowflake-Request-Id")
    if resp.status_code < 400:
        response = resp.json()
        request_id = response["request_id"]
        content = response["message"]["content"]
        # print(content)
        for item in content:
            if item["type"] == "sql":
                # print(item["statement"])
                df = pd.read_sql(item["statement"], connection)
                result_text = f"SQL Statement: {item['statement']}\n\nResults:\n{df.to_string(index=False)}"
                cortex_result = result_text
            elif item["type"] == "text":
                cortex_result = item["text"]
        return cortex_result
    else:
        raise Exception(
            f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
        )


def cortex_analyst_generate(state, writer: StreamWriter):
    """
    Use Cortex Analyst to lookup data within Snowflake

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates state with generated SQL results
    """
    print("---CORTEX ANALYST LOOKUP---")
    writer("Looking at data in Snowflake...")
    question = state.question
    resp = call_cortex_analyst(question)
    result = Document(page_content=resp)
    state.data.append(result)
    return {"data": state.data, "question": question}
