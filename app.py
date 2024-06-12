import os
import tempfile

import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.question_answering import load_qa_chain

import chromadb

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
retriever = vectorstore.as_retriever()

embedder = VideoIngester(
            chromadb_client,
            CHROMADB_COLLECTION
)


llm = Ollama(model="llama3")
rag_prompt = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template="""You answer questions about the contents of a transcribed audio file. 
                Use only the provided audio file transcription as context to answer the question. 
                Do not use any additional information.
                If you don't know the answer, just say that you don't know. Do not use external knowledge. 
                Make sure to reference your sources with quotes of the provided context as citations.
                \nQuestion: {question} \nContext: {context} \nAnswer:"""
                )
        )
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# load in qa_chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

st.set_page_config(page_title="WhisperVector")

def page():
    if len(st.session_state) == 0:

        st.session_state.chroma_query = ""
        st.session_state.chroma_response = ""
        st.session_state.file_uploader = []

        st.session_state.messages = []

    st.header("WhisperVector")

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
            segment_count = embedder.ingest_video_file(file_path)
        st.text(f"Ingested {segment_count} segments")
        os.remove(file_path)

    st.session_state["ingestion_spinner"] = st.empty()


    st.subheader("Collection Management")
    st.text(f"Collection: {CHROMADB_COLLECTION}")
    if st.button("Clear Collection"):
        chromadb_client.delete_collection(CHROMADB_COLLECTION)
        chromadb_client.get_or_create_collection(CHROMADB_COLLECTION)

    st.subheader("LLM Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("LLM Prompt"):

        with st.chat_message("user"):
            st.session_state.messages.append(
                {
                    'role': 'user',
                    'content': prompt,
                }
            )
            st.markdown(prompt)

        response = qa_chain.invoke(prompt)

        with st.chat_message("assistant"):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                 }
            )
            st.markdown(response)


if __name__ == "__main__":
    page()