import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
#from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.document_loaders import PyPDFLoader, TextLoader

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
)

def load_history(_):
    return memory.load_memory_variables({})["chat_history"]

@st.cache_data(show_spinner="Embedding file now...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    # loader = UnstructuredFileLoader(file_path)
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.session_state.api_key
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     Answer the question using ONLY the following context. If you don't know the answer,
     just say you don't know. DON'T make anything up.
     
     Context: {context}
     """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

with st.sidebar:
    st.markdown("""
        <small>
        🔗 <b>GitHub Repository</b><br>
        <a href="https://github.com/soohyunii/fullstackGPT.git" target="_blank">
        soohyunii/fullstackGPT
        </a>
        </small>
        """,
        unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    api_key = st.text_input("🔑 OpenAI API", placeholder="Input Your OpenAI API Key")
    submit = st.button("submit")
    st.markdown("<br>", unsafe_allow_html=True)
    


# api_key가 있어야 나머지 화면이 나타남 : 파일 업로드 + 채팅
if submit:
    if api_key:
        st.session_state.api_key = api_key
        st.sidebar.success("Checked API key.")
    else:
        st.sidebar.error("Please enter your API key.")
        

if not st.session_state.api_key == "":
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=st.session_state.api_key,
    )
    
    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt or .pdf file", 
            type=["pdf", "txt"]
        )
        
    if file:
        retriever = embed_file(file)
        send_message("ASK Anything!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file :)")

        if message:
            send_message(message, "human")
            try:
                chain = ({
                    "context":retriever | RunnableLambda(format_docs),
                    "chat_history": RunnableLambda(load_history),
                    "question": RunnablePassthrough()
                } | prompt | llm | StrOutputParser())
                
                answer = ""

                for chunk in chain.stream(message):
                    answer += chunk 
                
                memory.save_context(
                    {"question": message},
                    {"answer": answer}
                )
                send_message(answer, "ai")
                    
            except:
                st.error("❌ Invalid OpenAI API key. Please check your key and try again.")
                st.session_state["messages"] = []

else:
    st.session_state["messages"] = []            
    
# st.write(st.session_state["messages"])