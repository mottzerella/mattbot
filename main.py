__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import FireCrawlLoader
from langchain.docstore.document import Document as LCDocument # to avoid conflict with LlamaParse Document
import os
from huggingface_hub import login
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv, find_dotenv
from llama_index.readers.file import (DocxReader,EpubReader,HWPReader,ImageReader,IPYNBReader,MarkdownReader,MboxReader,PandasCSVReader,PandasExcelReader,PDFReader,PptxReader,VideoAudioReader)
from langchain_community.llms import DeepInfra
from langchain_community.chat_models import ChatDeepInfra
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import nest_asyncio
import streamlit as st

load_dotenv(find_dotenv(), override=True)

##OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
#PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
#LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
#LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# Huggingface Login Creds
#TOKEN = os.getenv('TOKEN')
#login(token=TOKEN)

#os.environ['token'] = TOKEN
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['LLAMA_CLOUD_API_KEY'] = st.secrets['LLAMA_CLOUD_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']

llama = "meta-llama/Meta-Llama-3.1-70B-Instruct"
#LLM = ChatOllama(model=llama, format='json', temperature = 0.5)
LLM = ChatOpenAI(model_name= 'gpt-4o', temperature = 0.5, top_p = 0.9) 
#Enable switching between Llama prompt and GPT-4o prompt 
#LLM = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct")
#LLM.model_kwargs = {'temperature': 0.5, 'repitition_penalty': 1.2,'max_new_tokens': 250, 'top_p': 0.9}



def chunk_data(data, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536) 
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


#PARSING DOCUMENTS

nest_asyncio.apply()
parser = LlamaParse(result_type="markdown")  # "markdown" and "text" are available
file_extractor = {".pdf": parser}
llama_parse_documents = SimpleDirectoryReader(input_files=['Matt_Zerella_Autobiography.pdf'], 
                                              file_extractor=file_extractor, 
                                              file_metadata=None).load_data()

context_document = ([x.to_langchain_format() for x in llama_parse_documents])

# CREATING CHUNKS AND VECTOR STORE

chunks = chunk_data(context_document)
vector_store = create_embeddings(chunks)

# PROMPTS

## GPT-4o Prompts

retriever_system_prompt = """You are a highly-intelligent individual designed to impersonate Matt Zerella.  Given the provided chat history, 
context document, and latest user question which might reference the chat history, formulate a standalone question which can be understood 
without the chat history.  Do not make up, extrapolate, or embellish information.  Do not answer the question, just reformulate if necessary
and return as is.
"""
retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", retriever_system_prompt),
            ("human", "{input}")])

system_prompt = """Your name is Matt Zerella, a relaxed individual who is excellent at answering job interview questions in a witty but professional way. 
The context provided is a document containing details about your life. 
Answer all questions in a brief manner with a casual tone and exceptional vocabulary, in way that positively reflects on your character.  
If you don't know the answer, just say that you don't know. Do not extrapolate, embellish, or make up information, and do not provide any unnecessary
information.  If the question is open-ended, ask for the user to make the question more specific.
{context}
{chat_history}
"""
main_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )


# SESSION STORE
def generate_chain(vector_store):

    llm = LLM
    retriever = vector_store.as_retriever(search_type='similarity', 
                                        search_kwargs={'k': 30})
    
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=retriever_prompt)

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=main_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)


    chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    return chain


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    #st.image('')
    st.subheader('Welcome to Mattbot!')
   #with st.sidebar:
    
    chain = generate_chain(vector_store)
    st.session_state.chain = chain

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    st.session_state.vs = vector_store
    question = st.text_input('What would you like to know about me?')

    if question: # if the user entered a question and hit enter
        if 'vs' in st.session_state and 'chain' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            chain = st.session_state.chain
            vector_store = st.session_state.vs
            #st.write(f'k: {k}')

            answer = chain.invoke({"input": question},
                        config={"configurable": {"session_id": '1234'}})

            
            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer['answer'])

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''
                st.success("Thanks for asking!")

            # the current question and answer
            value = f'Q: {question} \nA: {answer["answer"]}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)
