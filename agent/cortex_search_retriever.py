from typing import List

from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import os
from snowflake.core import Root
from snowflake.snowpark import Session

session = Session.builder.config("connection_name", "pm").getOrCreate()
root = Root(session)

# fetch service
my_service = (
    root.databases["JEFFHOLLAN_DEMO"]
    .schemas["SURVEY"]
    .cortex_search_services["DOCUMENTS"]
)


class CortexSearchRetriever(BaseRetriever):
    """A Snowflake Cortex Search retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """

    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # call the _aget_relevant_documents method synchronously
        resp = my_service.search(
            query=query,
            columns=["Document_Name", "Text"],
            # filter={"@eq": {"<column>": "<value>"}},
            limit=self.k,
        )
        return [
            Document(
                page_content=doc["Text"],
                metadata={"Document_Name": doc["Document_Name"]},
            )
            for doc in resp.to_dict()["results"]
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use

        Returns:
            List of relevant documents
        """
        # query service
        resp = my_service.search(
            query=query,
            columns=["Document_Name", "Text"],
            # filter={"@eq": {"<column>": "<value>"}},
            limit=self.k,
        )
        return [
            Document(
                page_content=doc["Text"],
                metadata={"Document_Name": doc["Document_Name"]},
            )
            for doc in resp.to_dict()["results"]
        ]
