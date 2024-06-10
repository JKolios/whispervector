import os
import tempfile
from pprint import pformat

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from videoingester import VideoIngester
from embedqueries import EmbedQueries

st.set_page_config(page_title="WhisperVector")



def submit_embed_query():
    st.session_state.chroma_response = st.session_state.query_interface.chroma_query(st.session_state.chroma_query)

def submit_llm_query():
    chain = st.session_state.llm | StrOutputParser()

    st.write(chain.invoke(st.session_state.llm_query))

def clear_collection():
    st.session_state.query_interface.clear_collection()

def page():
    if len(st.session_state) == 0:
        st.session_state.collection = "transcripts"
        st.session_state.embedder = VideoIngester(st.session_state.collection)

        st.session_state.query_interface = EmbedQueries(st.session_state.collection)
        vectorstore = Chroma(st.session_state.collection)
        st.session_state.chroma_query = ""
        st.session_state.chroma_response = ""
        st.session_state.file_uploader = []

        st.session_state.messages = []

    st.header("WhisperVector")

    st.subheader("Upload a video file")
    uploaded_files = st.file_uploader(
        "Upload document",
        type=["mp4", "mkv"],
        key=st.session_state.file_uploader,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            segment_count = st.session_state["embedder"].ingest(file_path, file.name)
        st.text(f"Ingested {segment_count} segments")
        os.remove(file_path)

    if st.button("Clear uploaded files"):
        st.session_state.file_uploader = []
        st.rerun()

    st.session_state["ingestion_spinner"] = st.empty()

    st.subheader("ChromaDB Queries")
    with st.form(key='query_form_chroma', clear_on_submit=True):
        st.text_input('Query:', value=st.session_state.chroma_query)
        st.form_submit_button('Submit', on_click=submit_embed_query)

    st.code(body=pformat(st.session_state.chroma_response, compact=False), language='json')

    st.subheader("Collection Management")
    st.text(f"Collection: {st.session_state.collection}")
    st.button("Clear Collection", on_click=clear_collection)

    st.subheader("LLM Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("LLM Prompt"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.session_state.messages.append(
                {
                    'role': 'user',
                    'content': prompt,
                }
            )
            response = ollama.chat(model='llama3', messages=st.session_state.messages)
            st.write(response['message']['content'])
            st.session_state.messages.append(
                {"role": "assistant",
                 "content": response['message']['content']
                 }
            )


if __name__ == "__main__":
    page()