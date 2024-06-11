import os
import tempfile
from pprint import pformat

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

import chromadb

from videoingester import VideoIngester
from embedqueries import EmbedQueries

CHROMADB_COLLECTION = "transcripts"

chromadb_client = chromadb.PersistentClient(path="./chromadb")

embedder = VideoIngester(
            CHROMADB_COLLECTION,
            chromadb_client
        )

query_interface = EmbedQueries(
            CHROMADB_COLLECTION,
            chromadb_client
        )

llm = Ollama(model="llama3")
chain = llm | StrOutputParser()

st.set_page_config(page_title="WhisperVector")

def clear_collection():
    query_interface.clear_collection()

def page():
    if len(st.session_state) == 0:

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
            segment_count = embedder.ingest(file_path, file.name)
        st.text(f"Ingested {segment_count} segments")
        os.remove(file_path)

    if st.button("Clear uploaded files"):
        st.session_state.file_uploader = []

    st.session_state["ingestion_spinner"] = st.empty()

    st.subheader("ChromaDB Queries")
    with st.form(key='query_form_chroma', clear_on_submit=True):
        st.text_input('Query:', value=st.session_state.chroma_query)
        submit_chroma_query = st.form_submit_button('Submit')

    if submit_chroma_query:
        st.write(query_interface.chroma_query(st.session_state.chroma_query))

    st.subheader("Collection Management")
    st.text(f"Collection: {CHROMADB_COLLECTION}")
    st.button("Clear Collection", on_click=clear_collection)

    st.subheader("LLM Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("LLM Prompt"):
        # Add user message to chat history
        # st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        # with st.chat_message("user"):
        #     st.markdown(prompt)

        with st.chat_message("user"):
            st.session_state.messages.append(
                {
                    'role': 'user',
                    'content': prompt,
                }
            )
            st.markdown(prompt)

        response = chain.invoke(prompt)

        with st.chat_message("assistant"):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                 }
            )
            st.markdown(response)
        #
        # with st.chat_message("assistant"):
        #     st.markdown(response)




if __name__ == "__main__":
    page()