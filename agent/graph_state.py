from typing import List

from typing_extensions import TypedDict
from langchain_core.documents.base import Document, Blob

from pydantic import BaseModel, Field
from typing import Union


class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        data: list of documents or retrieved data
    """

    question: str = Field(..., description="The question from the user")
    generation: str = Field(default="", description="The end generation from the agent")
    data: List[Union[Blob, Document]] = Field(
        default=[], description="The data relevant to help generate"
    )
    analysis_prompts: List[str] = Field(
        default=[],
        description="A list of prompts that can be used to help with analysis",
    )
    prompts_to_review: List[str] = Field(
        default=[], description="A list of prompts that need to be reviewed"
    )
