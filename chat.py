import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

import chromadb


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

def user_interface():
    if len(st.session_state) == 0:
        st.session_state.messages = []

    st.header("WhisperVector")

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
    user_interface()
