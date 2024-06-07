import os
import tempfile
import streamlit as st
from streamlit_chat import message
from textembeddingmaker import TextEmbeddingMaker
from embedqueries import EmbedQueries

st.set_page_config(page_title="WhisperVector")


def read_and_save_file():
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["embedder"].ingest(file_path)
        os.remove(file_path)
    st.session_state["file_uploader"] = []


def submit_embed_query():
    st.session_state.response = st.session_state.query_interface.query(st.session_state.query)



def page():
    if len(st.session_state) == 0:
        st.session_state.embedder = TextEmbeddingMaker()
        st.session_state.query_interface = EmbedQueries()
        st.session_state.query = ""
        st.session_state.response = ""

    st.header("WhisperVector")

    st.subheader("Upload a video file")
    st.file_uploader(
        "Upload document",
        type=["mp4", "mkv"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    with st.form(key='query_form', clear_on_submit=True, ):
        st.text_input('Query:', value=st.session_state["query"])
        st.form_submit_button('Submit', on_click=submit_embed_query)

    st.text_area('Response', value=st.session_state.response)

if __name__ == "__main__":
    page()