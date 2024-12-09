# Importsfrom 
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import FireCrawlLoader
from langchain.docstore.document import Document as LCDocument # to avoid conflict with LlamaParse Document
import os
from huggingface_hub import login
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from llama_index.readers.file import (DocxReader,EpubReader,HWPReader,ImageReader,IPYNBReader,MarkdownReader,MboxReader,PandasCSVReader,PandasExcelReader,PDFReader,PptxReader,VideoAudioReader)
from langchain_community.llms import DeepInfra
from langchain_community.chat_models import ChatDeepInfra
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path
from typing import List, Optional
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
import streamlit as st

os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['LLAMA_CLOUD_API_KEY'] = st.secrets['LLAMA_CLOUD_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']
parser = LlamaParse(result_type="markdown")

llama = "meta-llama/Meta-Llama-3.1-70B-Instruct"
#LLM = ChatOllama(model=llama, format='json', temperature = 0.5)
llm = ChatOpenAI(model_name= 'gpt-4o', temperature = 0.5, top_p = 0.9) 
#Enable switching between Llama prompt and GPT-4o prompt 
#LLM = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct")
#LLM.model_kwargs = {'temperature': 0.5, 'repitition_penalty': 1.2,'max_new_tokens': 250, 'top_p': 0.9}

nest_asyncio.apply()

@st.cache_resource
def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""
    file_extractor = {".pdf": parser}
    
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over a given paper.
    
        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=8,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
        
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query, 
        description=(f"useful for answering specific questions related to Matt Zerella's {name}")
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to Matt Zerella's {name}"
        ),
    )

    return vector_query_tool, summary_tool





agentic_system_prompt = """Your name is Matt Zerella, a relaxed individual who is excellent at answering job interview questions in a witty but professional way. 
The context provided is a document containing details about your life. 
Answer all questions in a brief manner with a kind, casual tone and exceptional vocabulary, in way that positively reflects on your character.  
If you don't know the answer, just say that you don't know, but that you would love to schedule an interview. Do not extrapolate, embellish, or make up information, and do not provide any unnecessary
information.  If the question is open-ended or too broad, ask for the user to make the question more specific.

If asked to share your contact info, share that your phone number is (408) 857-0815 and your email is mzerella2@gmail.com
If prompted to share your resume, produce a full list of chronological work experience with highlights for each role

{chat_history}
"""

prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("system", agentic_system_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
            ("human", "{input}")
        ]
    )



search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    description="A search engine optimized for comprehensive, accurate, and trusted results. Use when you need to answer questions about current events, or when you need to perform an internet search to find information that another tool can not provide. Input should be a search query.",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

langchain_tools = [search_tool]


papers = [
'Personal_life.pdf',
'answers_to_common_interview_questions.pdf',
'work_experience.pdf'
]
paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
multi_doc_tools=[t.to_langchain_tool() for t in initial_tools]

tools = multi_doc_tools + langchain_tools

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt,)

@st.cache_resource
def get_agent_executor(_agent, _tools):
    agent_executor = AgentExecutor(agent=agent, 
                                    tools=tools, 
                                    verbose=True, 
                                    return_intermediate_steps=True, 
                                    handle_parsing_errors=True, 
                                    max_iterations=10)
    return agent_executor
agent_executor = get_agent_executor(_agent=agent, _tools=tools)


def get_multi_doc_chain():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    multi_doc_chain = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        #output_messages_key="answer"
    )
    return multi_doc_chain

if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    #st.image('')
    st.header('Welcome to Mattbot! :party')
    st.subheader('I use an Agentic RAG workflow under the hood. Be friendly, I'm still in beta :smile')
   #with st.sidebar:
    
    chain = get_multi_doc_chain()
    st.session_state.chain = chain

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    #st.session_state.tools = tools
    #st.session_state.agent = agent
    question = st.text_input('What would you like to know about me?')

    if question: # if the user entered a question and hit enter
        if 'chain' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            #chain = st.session_state.chain
            #agent = st.session_state.agent
            #tools = st.session_state.tools
            #st.write(f'k: {k}')

            answer = chain.invoke({"input": question},
                        config={"configurable": {"session_id": '1234'}})

            
            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer['output'])

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''
                st.success("Thanks for asking!")

            # the current question and answer
            value = f'Q: {question} \nA: {answer["output"]}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)
