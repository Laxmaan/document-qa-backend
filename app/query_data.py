"""Create a ConversationalRetrievalChain for question/answering."""
import os
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI, AzureOpenAI
from langchain.vectorstores.base import VectorStore


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    # question_gen_llm = AzureOpenAI(
    #     deployment_name='idocgptmodels',
    #     openai_api_base="https://idocopenaigpt.openai.azure.com/",
    #     openai_api_key=os.environ['AZUREOPENAI_API_KEY'],
    #     # model="gpt-35-turbo",
    #     # engine='text-davinci-003',
    #     temperature=0,
    #     verbose=True,
    #     streaming=True,
    #     callback_manager=question_manager,
    # )
    # streaming_llm = AzureOpenAI(
    #     deployment_name='idocgptmodels',
    #     # model="gpt-35-turbo",
    #     streaming=True,
    #     callback_manager=stream_manager,
    #     verbose=True,
    #     temperature=0,
    # )

    question_gen_llm = OpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY_BASE'],
        streaming=True,
        callback_manager=question_manager,
        verbose=True,
        temperature=0,
    )

    streaming_llm = OpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY_BASE'],
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, 
        chain_type="stuff",
        prompt=QA_PROMPT,
        callback_manager=manager,
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True
    )
    return qa
