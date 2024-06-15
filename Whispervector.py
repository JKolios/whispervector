from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import chromadb
import streamlit as st

CHROMADB_COLLECTION = "transcripts"
DEFAULT_EMBEDDINGS_MODEL = "openai"

chromadb_client = chromadb.PersistentClient(path="../chromadb")
# ensure that the collection exists
chromadb_client.get_or_create_collection(CHROMADB_COLLECTION)

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
        client=chromadb_client,
        collection_name=CHROMADB_COLLECTION,
        embedding_function=embedding_function,
)


st.set_page_config(
    page_title="WhisperVector",
    page_icon="🤫",
)

st.markdown(
"""
    WhisperVector is an experiment with Retrieval-Augmented Generation using locally-hosted ML models.
"""
)

