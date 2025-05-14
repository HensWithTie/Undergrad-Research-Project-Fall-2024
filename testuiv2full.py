import os
import tempfile
import streamlit as st
from streamlit_chat import message
from pipelinev2 import ChatText
import time

st.set_page_config(page_title="ChatText")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    stime = time.time()
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    #st.session_state["user_input"] = ""

    #for file in st.session_state["file_uploader"]:
        # with tempfile.NamedTemporaryFile(delete=False) as tf:
        #     tf.write(file.getbuffer())
        #     file_path = tf.name

    #filename = "pokedict.json"
    filename = "Gen1.txt"
    #filename = "Gen1Sample.txt"
    #filename = "Jon.txt"
    with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {filename}"):
        st.session_state["assistant"].ingest(filename)
    #os.remove(file_path)
    etime = time.time()
    ttime = etime - stime
    print(f"total time: {ttime:.4f} seconds")


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatText()

    st.header("ChatText")

    #st.subheader("Upload a document")
    
    # st.file_uploader(
    #     "Upload document",
    #     type=["txt"],
    #     key="file_uploader",
    #     on_change=read_and_save_file,
    #     label_visibility="collapsed",
    #     accept_multiple_files=True,
    # )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    stime = time.time()
    st.text_input("Message", key="user_input", on_change=process_input)
    etime = time.time()
    ttime = etime - stime
    print(f"total response time: {ttime:.4f} seconds")

    if not st.session_state["assistant"].chain:
        read_and_save_file()


if __name__ == "__main__":
    print("page")
    page()