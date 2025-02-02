import sys
import os
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../agent")))


from agent import app as agent

st.title("Snowflake Intelligence Agent (TEST)")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
            yield message_chunk + "\n"
        elif (
            stream_mode == "messages"
            and isinstance(message_chunk[1], dict)
            and "langgraph_node" in message_chunk[1]
            and message_chunk[1]["langgraph_node"] == "generate"
        ):
            yield message_chunk[0].content


if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = langgraph_stream(prompt)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
