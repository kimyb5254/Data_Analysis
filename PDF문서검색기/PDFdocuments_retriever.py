import os
from dotenv import load_dotenv
from langchain_teddynote import logging

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages.chat import ChatMessage

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("pdf-rag-project", set_enable=True) # 추적 활성화

# 캐시 디렉토리 생성
if not os.path.exists(".cache"): #점은 숨김파일
    os.mkdir(".cache")
# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir('.cache/embeddings')


st.title('PDF기반 QA_test :)')

# 처음 1번만 실행, 대화 저장하기 위함
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'chain' not in  st.session_state:
    st.session_state['chain'] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button('대화초기화')
    uploaded_file = st.file_uploader('파일업로드', type=['pdf'])
    selected_model = st.selectbox('llm 선택', ["gpt-4o", 'gpt-4-turbo', 'gpt-4o-mini'], index=0)

# 이전 대화 포함하여 출력
def print_messages():
    for chat_message in st.session_state['messages']:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 매시지 추가
def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

#파일을 캐시 저장( 시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다.")

# 주어진 파일가지고 검색기 생성
def embed_file(file):
    # 인자로 받은 file 내용 읽어와서 저장(문자열 혹은 이진데이터로 반환)
    file_content = file.read()
    # 파일을 저장할 경로
    file_path = f'./.cache/files/{file.name}'
    # 파일을 쓰기모드로 열기
    with open(file_path, 'wb') as f:
        # 읽은 파일의 내용을 새로 연 파일에 쓰기
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever

def create_chain(retriever, model_name = selected_model): 

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
    # 메인파일의 위치를 따라서 프롬프트 폴더부터 소환
    prompt = load_prompt('PDFdocuments.yaml')
    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model = model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
##--------------------------------------------------------------------------------------------
# 파일 업로드 됐을 때 : 파일이 업로드 되어야 체인 생성됨
if uploaded_file:
    retriever = embed_file(uploaded_file) 
    chain = create_chain(retriever, model_name= selected_model)
    st.session_state['chain'] = chain

# 초기화 버튼이 눌리면
if clear_btn :
    st.session_state['messages'] = []

# 이전 대화 출력 : st.chat_message(chat_message.role).write(chat_message.content)
print_messages()

# 사용자의 입력
warning_msg = st.empty()

# 사용자의 입력
user_input = st.chat_input("궁금한 것을 물어보세요!")

# 사용자 입력 들어오면
if user_input:

    # 체인 생성
    chain = st.session_state['chain']

    if chain is not None:
        # 사용자 입력 쓰기
        st.chat_message('user').write(user_input)
        # 스트리밍 출력하기
        response = chain.stream(user_input)

        with st.chat_message('assistant'):
            # 답변 한꺼번에 나중에 나오는 거 말고 토큰별로 완료되는대로 출력하기
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
        # 새로운 대화 저장
        add_message('user', user_input)
        add_message('assistant',ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error('파일을 업로드해주세요')