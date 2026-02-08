import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import json

response = None

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        # string looks like json => json
        text = text.replace("```","").replace("json","")
        return json.loads(text)

output_parser = JsonOutputParser()

if "api_key" not in st.session_state:
    st.session_state.api_key = ""


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
            You are a helpful assistant that is role playing as a teacher.
            Based ONLY on the following context make 10 questions to test the user's
            Knowledge about the text.
            
            Each question should have 4 answers, three of them must be incorrect and 
            one should be correct.
            
            If context language setting, You MUST make it by following language.

            Difficulty mode: {mode}

            Difficulty rules:
            - easy: direct facts from the text, simple wording, obvious correct answer, weak distractors.
            - hard: deeper understanding, inference/comparison, more specific details, plausible distractors.

                        
            Use (O) to signal the correct answer.
            
            Question examples:
            
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(O)
            
            Question: What is the capital of the Georgia?
            Answers: Baku|Tbilisi(O)|Manila|Beirut
            
            Question: What was the Avatar released?
            Answers: 2007|2001|2009(O)|1998
            
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(O)|Painter|Actor|Model
            
            Your turn!
            
            Context: {context}
         """
        )
    ])


formatting_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        """
        You are a powerful formatting algorithm.

        You format exam questions into JSON format.
        Answers with (o) are the correct ones.

        Example Input:
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)

        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut

        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998

        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model


        Example Output:

        ```json
        {{ 
        "questions": [
            {{
            "question": "What is the color of the ocean?",
            "answers": [
                {{
                "answer": "Red",
                "correct": false
                }},
                {{
                "answer": "Yellow",
                "correct": false
                }},
                {{
                "answer": "Green",
                "correct": false
                }},
                {{
                "answer": "Blue",
                "correct": true
                }},
                ]
            }},
            {{
            "question": "What is the capital or Georgia?",
            "answers": [
                {{
                "answer": "Baku",
                "correct": false
                }},
                {{
                "answer": "Tbilisi",
                "correct": true
                }},
                {{
                "answer": "Manila",
                "correct": false
                }},
                {{
                "answer": "Beirut",
                "correct": false
                }},
                ]
            }},
            {{
            "question": "When was Avatar released?",
            "answers": [
                {{
                "answer": "2007",
                "correct": false
                }},
                {{
                "answer": "2001",
                "correct": false
                }},
                {{
                "answer": "2009",
                "correct": true
                }},
                {{
                "answer": "1998",
                "correct": false
                }},
                ]
            }},
            {{
            "question": "Who was Julius Caesar?",
            "answers": [
                {{
                "answer": "A Roman Emperor",
                "correct": true
                }},
                {{
                "answer": "Painter",
                "correct": false
                }},
                {{
                "answer": "Actor",
                "correct": false
                }},
                {{
                "answer": "Model",
                "correct": false
                }},
                ]
            }}
            ]
        }}
        ```
        Your turn!
        Questions: {context}
        """ 
    )
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
    


# api_key가 있어야 나머지 화면이 나타남 : 파일 업로드 or 검색
if submit:
    if api_key:
        st.session_state.api_key = api_key
        st.sidebar.success("Checked API key.")
    else:
        st.sidebar.error("Please enter your API key.")
        

if not st.session_state.api_key == "":
    
    
    @st.cache_data(show_spinner="Searching Wikipedia....")
    def wiki_search(topic):
        retriever = WikipediaRetriever(top_k_results=3, lang="en")
        docs = retriever.get_relevant_documents(topic)
        return docs


    @st.cache_data(show_spinner="Loading file now...")
    def split_file(file):
        file_content = file.getvalue()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        
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
        return docs


    @st.cache_data(show_spinner="Making Quiz......")
    def run_quiz_chain(_docs, _mode, _api_key):
        llm_local = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo-1106",
            streaming=False,  # cache 함수 안에서는 streaming 끄는 게 안전
            openai_api_key=_api_key,
        )
        formatting_chain = formatting_prompt | llm_local
        question_chain = {
            "context": RunnableLambda(lambda x: format_docs(x["docs"])),
            "mode": RunnableLambda(lambda x: x["mode"]),
        } | questions_prompt | llm_local

        chain = {"context": question_chain} | formatting_chain | output_parser
        response = chain.invoke({"docs": _docs, "mode": _mode})
        return response
    
    
    with st.sidebar:
        docs = None
        choice = st.selectbox(
            "Choose what you want to use.", 
            (
                "File", 
                "Wikipedia Article",
            )
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .txt or .pdf file",
                type=["pdf", "txt"]
            )
            # we don't need vectorstore
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)
        
        
        mode = st.radio("Choose Quiz Mode", ["easy", "hard"], horizontal=True)

        
        if docs:
            if st.button("✅ Generate Quiz", use_container_width=True):
                response = run_quiz_chain(docs, mode, st.session_state.api_key)
                
            
    if not docs:
        st.markdown(
        """
        Quiz GPT
        
        1️⃣ Input Your API KEY
        
        2️⃣ Get started by uploading a file or searching on Wikipedia in the sidebar.
        
        3️⃣ Click 'Generate' Button
        
        """
    )
            
else:
    st.markdown(
        """
        Quiz GPT
        
        1️⃣ Input Your API KEY
        
        2️⃣ Get started by uploading a file or searching on Wikipedia in the sidebar.
        
        3️⃣ Click 'Generate' Button
        """
    )



 
if response:
    with st.form("questions_form"):  # 'form' only show the results when you submit button!
        for idx, question in enumerate(response["questions"]):
            st.write(question["question"])
            value = st.radio(
                "", 
                [answer["answer"] for answer in question["answers"]], 
                index=None,
                key=f"question_{idx}"
            )
            if {"answer":value, "correct":True} in question["answers"]:
                st.success("Correct")
            elif value is not None:
                st.error("Wrong")
        button = st.form_submit_button()
    
    