from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        data: list of documents or retrieved data
    """

    question: str
    generation: str
    data: List[str]
    analysis_prompts: List[str]
