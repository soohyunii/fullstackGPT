# 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
# 파일 업로드 및 채팅 기록을 구현합니다.
# 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.

# 코드를 공개 Github 리포지토리에 푸시합니다.
# 단. OpenAI API 키를 Github 리포지토리에 푸시하지 않도록 주의하세요.
# 여기에서 계정을 개설하세요: https://share.streamlit.io/
# 다음 단계를 따르세요: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1
# 앱의 구조가 아래와 같은지 확인하고 배포 양식의 Main file path 에 app.py를 작성하세요.
# your-repo/
# ├── .../
# ├── app.py
# └── requirements.txt
# 과제 제출 링크는 반드시 streamlit.app URL 이 되도록 하세요.

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

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
    loader = UnstructuredFileLoader(file_path)
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
    st.markdown("🔗 GitHub Repository")
    st.markdown(
        "[👉 View Source Code](https://github.com/your-username/your-repo)"
    )
    api_key = st.text_input("🔑 OpenAI API", placeholder="Input Your OpenAI API Key")
    submit = st.button("submit")
    


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
            chain = ({
                "context":retriever | RunnableLambda(format_docs),
                "chat_history": RunnableLambda(load_history),
                "question": RunnablePassthrough()
            } | prompt | llm | StrOutputParser())
            with st.chat_message("ai"):
                placeholder = st.empty()
                answer = ""

                for chunk in chain.stream(message):
                    answer += chunk
                    placeholder.markdown(answer)

                memory.save_context(
                    {"question": message},
                    {"answer": answer}
                )
                save_message(answer, "ai")
                    

else:
    st.session_state["messages"] = []            
    
# st.write(st.session_state["messages"])