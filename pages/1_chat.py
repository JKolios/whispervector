import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


from Whispervector import vectorstore


retriever = vectorstore.as_retriever()


# llm = Ollama(model="llama3-chatqa:8b")
llm = Ollama(model="llama3")

rag_prompt = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template="""You answer questions about the contents of a PDF text file. 
                Use only information from that text file to answer questions. 
                Do not use any additional information.
                If you don't know the answer, just say that you don't know. Do not use external knowledge. 
                Make sure your answers are as exhaustive as possible.
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


def user_interface():
    if not st.session_state.get('messages'):
        st.session_state.messages = []

    st.header("WhisperVector")

    st.subheader("Chat with RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("LLM Prompt"):

        with st.chat_message("user", avatar="🧑"):
            st.session_state.messages.append(
                {
                    'role': 'user',
                    'content': prompt,
                    'avatar': "🧑"
                }
            )
            st.markdown(prompt)

        rag_response = qa_chain.invoke(prompt)
        base_llm_response = llm.invoke(prompt)

        with st.chat_message("assistant", avatar="📚️"):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": rag_response,
                    'avatar': "📚"
                 }
            )
            st.markdown(rag_response)

        with st.chat_message("assistant", avatar="🤖"):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": base_llm_response,
                    'avatar': "🤖"
                 }
            )
            st.markdown(base_llm_response)


if __name__ == "__main__":
    user_interface()
