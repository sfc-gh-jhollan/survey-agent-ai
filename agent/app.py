from langgraph.graph import END, StateGraph, START

import cortex_analyst_retriever
from graph_flow import retrieve, route_question, generate
from graph_state import GraphState

workflow = StateGraph(GraphState)
workflow.add_node("snowflake_store", cortex_analyst_retriever.cortex_analyst_generate)
workflow.add_node("doc_store", retrieve)
workflow.add_node("generate", generate)
# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "snowflake_store": "snowflake_store",
        "doc_store": "doc_store",
    },
)
workflow.add_edge("snowflake_store", "generate")
workflow.add_edge("doc_store", "generate")
workflow.add_edge("generate", END)


# Compile
app = workflow.compile()

from pprint import pprint

# Run
inputs = {"question": "How much are we spending on each survey communication type?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")

# Final generation
pprint(value["generation"])
