import os
import tempfile

import streamlit as st

import chromadb

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import numpy as np
from sklearn.manifold import TSNE

import plotly.express as px

from videoingester import VideoIngester

CHROMADB_COLLECTION = "transcripts"

chromadb_client = chromadb.PersistentClient(path="./chromadb")
# ensure that the collection exists
chromadb_client.get_or_create_collection(CHROMADB_COLLECTION)

vectorstore = Chroma(
        client=chromadb_client,
        collection_name=CHROMADB_COLLECTION,
        embedding_function=OllamaEmbeddings(model='llama3'),
)

ingester = VideoIngester(
            chromadb_client,
            CHROMADB_COLLECTION
)


def plot_embeddings(embeddings, documents):
    n_components = 3  # 3D
    embs = np.array(embeddings)  # converting to numpy array
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=5)
    reduced_vectors = tsne.fit_transform(embs)

    reduced_vectors = np.c_[reduced_vectors, documents]
    # Create a 3D scatter plot
    fig = px.scatter_3d(reduced_vectors,  x=0, y=1, z=2, hover_data=[3])
    return fig


def user_interface():
    if len(st.session_state) == 0:
        st.session_state.file_uploader = []

    st.subheader("Ingest video files")

    with st.form("file-uploader", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload document",
            type=["mp4", "mkv"],
            key=st.session_state.file_uploader,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        ingested = st.form_submit_button("Ingest Files")

    if ingested and uploaded_files is not None:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                segment_count = ingester.ingest_video_file(file_path)
            st.text(f"Ingested {segment_count} segments")
            os.remove(file_path)

    st.session_state["ingestion_spinner"] = st.empty()

    st.subheader("Collection")

    st.text(f"Collection: {CHROMADB_COLLECTION}")
    if st.button("Clear Collection"):
        chromadb_client.delete_collection(CHROMADB_COLLECTION)
        chromadb_client.get_or_create_collection(CHROMADB_COLLECTION)

    embeddings = chromadb_client.get_or_create_collection(CHROMADB_COLLECTION).get(include=['embeddings', 'documents'])
    if embeddings['ids']:
        st.plotly_chart(plot_embeddings(embeddings['embeddings'], embeddings['documents']), use_container_width=False)


if __name__ == "__main__":
    user_interface()
