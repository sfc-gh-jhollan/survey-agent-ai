from langgraph.graph import END, StateGraph, START
import argparse

import cortex_analyst_retriever
from graph_flow import (
    retrieve,
    route_question,
    generate,
    generate_analysis_prompts,
    exec_sql_analysis,
    analyze_results,
)
from graph_state import GraphState

parser = argparse.ArgumentParser(description="Run the survey agent.")
parser.add_argument(
    "-q",
    "--question",
    type=str,
    default="How much are we spending on each survey communication type?",
    help="Question to ask the survey agent",
)
args = parser.parse_args()


workflow = StateGraph(GraphState)
workflow.add_node("snowflake_store", cortex_analyst_retriever.cortex_analyst_generate)
workflow.add_node("doc_store", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("generate_analysis_leads", generate_analysis_prompts)
workflow.add_node("exec_sql_analysis", exec_sql_analysis)
workflow.add_node("analyze_results", analyze_results)
# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "snowflake_store": "snowflake_store",
        "doc_store": "doc_store",
        "data_analysis": "generate_analysis_leads",
    },
)
workflow.add_edge("snowflake_store", "generate")
workflow.add_edge("doc_store", "generate")
workflow.add_edge("generate_analysis_leads", "exec_sql_analysis")
workflow.add_edge("exec_sql_analysis", "analyze_results")
workflow.add_edge("analyze_results", "generate")

workflow.add_edge("generate", END)


# Compile
app = workflow.compile()

from pprint import pprint

# Run
inputs = {"question": args.question}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")

# Final generation
# if generation exists in value, print generate. If not, just print value
if "generation" in value:
    pprint(value["generation"])
else:
    pprint(value)
