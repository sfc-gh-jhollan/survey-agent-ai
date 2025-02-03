import sys
import os
import streamlit as st
from openai import OpenAI


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../agent")))


from agent import app as agent

st.title("Snowflake Intelligence Agent (TEST)")
chain_of_thought = st.empty()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


if "messages" not in st.session_state:
    st.session_state.messages = []

if "status_updates" not in st.session_state:
    st.session_state.status_updates = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def langgraph_stream(prompt):
    inputs = {"question": prompt}
    for stream_mode, *chunk in agent.app.stream(
        inputs, stream_mode=["messages", "custom"]
    ):
        message_chunk = chunk[0]

        if stream_mode == "custom":
            st.session_state.status_updates.append(message_chunk)
            # loop through all messages in status_updates. Convert into markdown. The first in list (index 0) should be at top, then a line break, then a simple checkmark and the text for the next item, and continue until a full string is generated in markdown
            chain_of_thought.update(
                label="\n\n".join(
                    [f"- {message}" for message in st.session_state.status_updates]
                )
            )
        elif (
            stream_mode == "messages"
            and isinstance(message_chunk[1], dict)
            and "langgraph_node" in message_chunk[1]
            and message_chunk[1]["langgraph_node"] == "generate"
        ):
            yield message_chunk[0].content.replace("\n", "\n\n").replace("$", "\\$")


if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chain_of_thought = st.status("Thinking...")
        if (
            prompt
            == "Draft an email to send to my team on the opportunity and next steps"
        ):
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        else:
            stream = langgraph_stream(prompt)
            response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
    if prompt == "Show the weekly snapshot of total costs":
        import pandas as pd

        df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../data/cost_by_week.csv")
        )
        st.line_chart(df, x="Date", y="Cost")

    # delete chain_of_thought
    chain_of_thought.update(label="Complete", expanded=False, state="complete")
    st.session_state.status_updates = []
