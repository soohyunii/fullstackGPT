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


@st.cache_data(show_spinner="Embedding file now...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path)
    

with st.sidebar:
    api_key = st.text_input("🔑 OpenAI API", placeholder="Input Your OpenAI API Key")
    submit = st.button("submit")

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    openai_api_key=api_key
)

# api_key가 있어야 나머지 화면이 나타남 : 파일 업로드 + 채팅
if submit:
    if api_key:
        with st.sidebar:
            file = st.file_uploader(
                "Upload a .txt or .pdf file", 
                type=["pdf", "txt"]
            )
    else:
        st.sidebar.error("Please enter your API key.")
        

if file:
    retriever = embed_file(file)