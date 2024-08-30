from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 이메일 본문으로부터 주요 엔티티 추출
class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    phone_number : str = Field(description="메일을 보낸 사람의 전화번호")
    company: str = Field(description="메일을 보낸 사람의 회사 정보")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    Meeting_date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간. 언급이 안되면 공백")


load_dotenv()

st.title('Email Summary :)')

# 처음 1번만 실행, 대화 저장하기 위함
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button('대화초기화')

# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state['messages']:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 매시지 추가
def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

def create_email_parsing_chain():
    # PydanticOutputParser 생성
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    prompt = PromptTemplate.from_template(
        """
    You are a helpful assistant. Please answer the following questions in KOREAN.

    #QUESTION:
    다음의 이메일 내용 중에서 주요 내용을 추출해 주세요.

    #EMAIL CONVERSATION:
    {email_conversation}

    #FORMAT:
    {format}
    """
    )

    # format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
    prompt = prompt.partial(format=output_parser.get_format_instructions())

    # 체인 생성(어려움 기준으로 파서부터 작성)
    chain = prompt | ChatOpenAI(model='gpt-4o', temperature=0) |output_parser

    return chain

def create_report_chain():
    prompt = load_prompt('summary_report_format.yaml')
    output_parser = StrOutputParser()
    chain = prompt |ChatOpenAI(model='gpt-4o', temperature=0)| output_parser
    return chain

# 초기화 버튼이 눌리면
if clear_btn :
    st.session_state['messages'] = []

# 이전 대화 출력 
print_messages()

# 사용자의 입력
user_input = st.chat_input("이메일 전체 항목을 넣어주세요")

#만약 사용자 입력이 들어오면
if user_input:
    # 사용자 입력
    st.chat_message('user').write(user_input)

    # 1) 이메일을 파싱하는 체인 생성 및 실행
    email_chain = create_email_parsing_chain()
    answer = email_chain.invoke({'email_conversation':user_input})

    # 2) 이메일 요약 리포트 생성
    report_chain = create_report_chain()
    report_chain_input = {
        "sender": answer.person,
        "company": answer.company,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "Meeting_date": answer.Meeting_date,
    }

    response = report_chain.stream(report_chain_input) 
    with st.chat_message('assistant'):
        # 답변 한꺼번에 나중에 나오는 거 말고 토큰별로 완료되는대로 출력하기
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화내용 저장
    add_message('user', user_input)
    add_message('assistant',ai_answer)