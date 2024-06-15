import os
import tempfile

import streamlit as st

import numpy as np
from sklearn.manifold import TSNE

import plotly.express as px

from ingesters.videoingester import VideoIngester
from ingesters.textingester import TextIngester, PDFIngester

from Whispervector import chromadb_client, CHROMADB_COLLECTION, vectorstore



ALL_DOC_TYPES = ["mp4", "mkv", ".txt", ".pdf"]


def ingest_file(file_path, original_filename):
    _, file_extension = os.path.splitext(original_filename)
    if file_extension in ['.txt']:
        ingester = TextIngester(vectorstore)
    elif file_extension in ['.mkv', '.mp4']:
        ingester = VideoIngester(vectorstore)
    elif file_extension in ['.pdf']:
        ingester = PDFIngester(vectorstore)
    else:
        raise ValueError('Unsupported file type for ingestion')
    ingester.ingest_file(file_path)


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
    if not st.session_state.get('file_uploader'):
        st.session_state.file_uploader = []

    st.subheader("Ingest video files")

    with st.form("file-uploader", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload document",
            type=ALL_DOC_TYPES,
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
                ingest_file(file_path, file.name)
            st.text(f"Ingested {file}")
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
