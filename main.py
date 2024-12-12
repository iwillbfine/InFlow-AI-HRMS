import os  # 운영체제와 관련된 경로 및 환경 변수 작업을 위한 라이브러리
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 로드하기 위한 라이브러리
from fastapi import FastAPI, HTTPException  # FastAPI 서버 및 HTTP 예외 처리
from fastapi.responses import JSONResponse  # JSON 응답을 반환하기 위한 클래스
from fastapi.middleware.cors import CORSMiddleware  # CORS 설정을 위한 미들웨어
from contextlib import (
    asynccontextmanager,
)  # 비동기 리소스 관리에 유용한 컨텍스트 매니저
from pydantic import BaseModel  # 데이터 유효성 검증을 위한 모델 클래스

# LangChain 관련 모듈
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)  # 문서를 로드하기 위한 로더
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # 긴 텍스트를 분할하기 위한 유틸리티
from langchain_community.vectorstores import Chroma  # 벡터 스토어로 Chroma를 사용
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)  # OpenAI 기반 임베딩 및 LLM 사용

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)  # 검색된 문서 결합 체인 생성
from langchain.chains import (
    create_history_aware_retriever,
)  # 대화 히스토리를 고려한 검색기 생성
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # 대화형 프롬프트 템플릿 정의
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)  # 대화 히스토리를 처리하는 실행 가능 체인
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)

# 대화 메시지 히스토리 관리
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# 배치 관련 라이브러리
from sqlalchemy import create_engine
from sqlalchemy.sql import text  # SQLAlchemy의 text 모듈 추가
import pandas as pd
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor  # 병렬 처리를 위한 모듈
from datetime import datetime  # 배치 주기를 위한 모듈
import shutil  # 디렉토리 삭제를 위한 모듈
import asyncio  # 시간 대기를 위한 모듈
import re  # 단어 검색시 정규표현식 사용
from fasteners import InterProcessLock

import traceback  # 에러 디버깅 및 트레이스백 추적
import uvicorn  # FastAPI 서버 실행


# 로깅 설정
import time
import logging

# claude 임베딩
from langchain_voyageai import VoyageAIEmbeddings

# claude LLM
from langchain_anthropic import ChatAnthropic

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# .env 파일 로드
load_dotenv()

# # 데이터 경로 설정
# DATA_DIR = r"C:\lecture\FinalProject\InFlow-AI\data"
# CHROMA_DB_DIR = f"C:/lecture/FinalProject/InFlow-AI/chromadb_worker_{os.getpid()}"
# CSV_DIR = r"C:\lecture\FinalProject\InFlow-AI\chromadb\csv_files"


# 데이터 경로 설정
# DATA_DIR = r"C:\lecture\FinalProject\InFlow-AI\data"
# CHROMA_DB_DIR = f"C:/lecture/FinalProject/InFlow-AI/chromadb_worker_{os.getpid()}"
# CSV_DIR = r"C:\lecture\FinalProject\InFlow-AI\chromadb\csv_files"
# LOCK_FILE = r"C:/lecture/FinalProject/InFlow-AI/chroma.lock"  # 잠금 파일 경로

# # ec2 데이터 경로
DATA_DIR = r"/home/ec2-user/InFlow-AI/data"
CHROMA_DB_DIR = f"/home/ec2-user/InFlow-AI/chromadb_worker_{os.getpid()}"
CSV_DIR = r"/home/ec2-user/InFlow-AI/chromadb/csv_files"
LOCK_FILE = r"/home/ec2-user/InFlow-AI/chroma.lock"  # 잠금 파일 경로

# MariaDB 연결 정보
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "test_db")

# OpenAI API 키 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOAGE_API_KEY = os.getenv("VOAGE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# OpenAI 임베딩 설정
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)

# # claude 임베딩 모델
# embeddings = VoyageAIEmbeddings(
#     voyage_api_key=VOAGE_API_KEY, model="voyage-lite-02-instruct"
# )


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 벡터 스토어 초기화
vectorstore = None


# CSV 디렉토리와 Chroma DB 디렉토리를 초기화 및 생성
# def initialize_directories():
#     lock = InterProcessLock(LOCK_FILE)
#     if lock.acquire(blocking=False):  # 잠금 획득 시만 초기화 수행
#         try:
#             if os.path.exists(CHROMA_DB_DIR):
#                 shutil.rmtree(CHROMA_DB_DIR)
#                 print(f"Cleared Chroma DB directory: {CHROMA_DB_DIR}")
#             os.makedirs(CHROMA_DB_DIR, exist_ok=True)
#             print(f"Created Chroma DB directory: {CHROMA_DB_DIR}")
#         finally:
#             lock.release()
#     else:
#         print("Another process is already initializing directories. Skipping...")


def initialize_directories():
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)  # 잠금 파일의 디렉토리 생성
    lock = InterProcessLock(LOCK_FILE)  # 잠금 파일을 통한 동시성 제어

    try:
        if lock.acquire(blocking=True, timeout=10):  # 잠금 획득 대기 (최대 10초)
            try:
                if os.path.exists(CHROMA_DB_DIR):
                    print(f"Chroma DB directory already exists: {CHROMA_DB_DIR}")
                else:
                    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
                    print(f"Created Chroma DB directory: {CHROMA_DB_DIR}")
            finally:
                lock.release()
        else:
            print("Failed to acquire lock. Another process is initializing.")
    except Exception as e:
        print(f"Error during directory initialization: {e}")


# SQLAlchemy 엔진 생성(mariadb )
def connect_sqlalchemy():
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        return engine
    except Exception as e:
        print(f"Error creating SQLAlchemy engine: {e}")
        raise


# CSV.pdf 데이터를 벡터화하고 Chroma DB에 저장
def store_files_in_chroma():
    """CSV와 PDF 파일을 벡터 스토어에 저장"""
    global vectorstore
    documents = []

    # PDF 파일 처리
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load_and_split())

    # 문서 분할 및 벡터화
    split_docs = text_splitter.split_documents(documents)
    if split_docs:
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
        print("Data stored in Chroma DB.")


# 24시간 간격으로 배치를 실행
async def run_batch():

    while True:
        try:
            print(f"Batch started at {datetime.now()}")
            initialize_directories()  # 1. 디렉토리 초기화 및 생성
            # export_tables_to_csv()  # 2. 모든 테이블 데이터를 CSV로 저장
            store_files_in_chroma()  # 3. CSV, PDF 데이터를 벡터 DB에 저장
            print(f"Batch completed at {datetime.now()}")
        except Exception as e:
            print(f"Error during batch execution: {e}")
        await asyncio.sleep(24 * 60 * 60)  # 4. 24시간 대기


# FastAPI 인스턴스 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# FastAPI 시작 시 초기화 작업
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan 이벤트 핸들러"""
    global vectorstore  # 전역 변수로 vectorstore 사용

    try:
        close_vectorstore()  # Vectorstore 종료
        print("Initializing directories...")
        initialize_directories()

        # print("Exporting tables to CSV...")
        # export_tables_to_csv()

        print("Storing files in Chroma...")
        store_files_in_chroma()

        print("Chroma Vectorstore Initialized.")

        # 24시간 배치 작업 병렬 실행
        asyncio.create_task(run_batch())
        yield  # 애플리케이션이 실행될 동안 이벤트 처리

    except Exception as e:
        print(f"Error during startup: {e}")
        raise

    finally:
        print("Shutting down the application...")
        # 필요시 클린업 작업 추가


# Vectorstore 작업 종료 함수
def close_vectorstore():
    global vectorstore
    if vectorstore is not None:
        vectorstore = None
        print("Vectorstore closed successfully.")


# Lifespan 이벤트 핸들러
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan 이벤트 핸들러"""
    try:
        # 애플리케이션 초기화 시 실행할 작업
        print("Application starting...")
        initialize_directories()  # 디렉토리 초기화
        # export_tables_to_csv()  # CSV 생성
        store_files_in_chroma()  # Vectorstore에 데이터 저장
        yield  # 애플리케이션 실행
    finally:
        # 애플리케이션 종료 시 실행할 작업
        print("Shutting down...")
        close_vectorstore()  # Vectorstore 종료


# FastAPI 인스턴스에 lifespan 설정
app = FastAPI(lifespan=lifespan)


# 입력 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str
    employee_id: str
    session_id: str


# 답변 문서 직렬화 및 테이블 설명 포함
def serialize_documents(documents):
    serialized_docs = []
    for doc in documents:
        serialized_docs.append(
            {
                "page_content": doc.page_content.replace("\\", ""),
                "metadata": {
                    "table": doc.metadata.get("table", "unknown"),
                    "table_korean": doc.metadata.get(
                        "table_korean", "알 수 없는 테이블"
                    ),
                },
            }
        )
    return serialized_docs


def serialize_data(data):
    """
    모든 데이터에서 NaN, Infinity 값을 안전한 값으로 변환하고 Timestamp 객체를 문자열로 변환
    """
    if isinstance(data, list):
        return [serialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}
    elif isinstance(data, pd.Timestamp):  # pandas Timestamp 객체 처리
        return data.isoformat()
    elif hasattr(data, "isoformat"):  # 일반 datetime 객체 처리
        return data.isoformat()
    elif isinstance(data, float):  # NaN, Infinity 처리
        if pd.isna(data) or data != data:  # NaN 확인
            return None
        if data == float("inf") or data == float("-inf"):  # Infinity 확인
            return None
        return data
    else:
        return data


# 히스토리 생성 및 조회

chat_history_storage = {}  # 대화 히스토리 저장소


def get_or_create_history(session_id):
    if session_id not in chat_history_storage:
        chat_history_storage[session_id] = ChatMessageHistory()
    return chat_history_storage[session_id]


# 체인 생성 함수
def create_chain_with_message_history():
    global rag_chain  # 전역 변수를 사용하도록 선언

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"lambda_mult": 0.95, "fetch_k": 4, "k": 3}
    )

    # 질문 문맥화 프롬프트
    contextualize_q_system_prompt = """

    Given a chat history, the latest user question, employee summary information (included in the first history message), \
    and metadata about the HR management system tables, formulate a standalone question. \
    The standalone question must be understandable without the chat history and \
    should utilize the given employee summary and table metadata if relevant. 
    
    **Do NOT answer the question. Only generate or reformulate the question.**
    
    Instructions:
    0. 절대 질문에 답하지마
    1. Use the employee summary only when the user question explicitly references employee-specific data (e.g., department, vacation, attendance).
    2. Do not assume implicit references unless the user question clearly relates to the employee summary.
    3. Reformulate the user question as an **informational request** rather than a procedural or operational query. 
    - For example, use phrasing like "I want to know about..." instead of "How can I retrieve...".
    4. If the question is ambiguous, reformulate it to match the most plausible intent without over-specifying details from the employee summary.
    5. Only use explicit data from the employee summary and context to enhance the question if directly relevant.

    질문은 반드시 한국어 존댓말로 해줘.
    **Do NOT answer the question. Only generate or reformulate the question.**

    Reformulate the given sentence or phrase to make it resemble a clear and standalone question as closely as possible."
    """
    # 예상 질문안에 근퇴 or 출퇴근/ 재택 / 초과 근무 / 휴가 / 휴직 / 복직 / 출장 / 파견 / 급여 / 계약 or 증명 / 부서 / 평가
    #
    #

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문 답변 프롬프트
    qa_system_prompt = """
    You are an advanced HR management system chatbot.

    Your task is to provide accurate, polite, and helpful answers in Korean using the \
    retrieved context, employee summary information, and HR table metadata.

    Instructions:
    1. Use the retrieved context to answer the question accurately.
    2. If the question references employee-specific data (e.g., vacation, attendance), \
    utilize the provided employee summary information in the answer.
    3. If the question involves HR system functionalities or policies, include \
    relevant HR table metadata (e.g., table names, descriptions) in the explanation.
    4. If no clear answer can be provided:
    - Guess the user’s intent based on the context and give a relevant suggestion.
    - If no reasonable inference can be made, respond politely:
        "질문에 대해 정확한 답변을 드리기 어려워요. 조금 더 구체적으로 질문해 주시면 감사하겠습니다." 
    
    Guidelines:
    - Write in polite Korean (존댓말).
    - Make the response conversational and user-friendly.
    - Avoid including unnecessary technical details unless requested.

    Example:
    - Question: "How many vacation days do I have left?"
    Context: Vacation Table Metadata, Employee Summary
    Response: "현재 사용 가능하신 연차는 10일이고, 병가는 2일 남아 있습니다."

    - Question: "How is my department structured?"
    Context: Department Table Metadata, Employee Summary
    Response: "사용자님이 속한 부서는 'oo부'입니다. 부서 구성원은 총 12명으로 구성되어 있으며, 세부 정보는 다음과 같습니다: 홍길동(팀원), 이준석(팀원), ..."
    
    반드시 한국어 존댓말로 답변해줘.
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # # LLM 설정
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.1,
    )

    # LLM 설정
    # llm = ChatAnthropic(
    #     api_key=CLAUDE_API_KEY, model="claude-3-haiku-20240307", temperature=0.1
    # )

    contextualize_chain = RunnableWithMessageHistory(
        runnable=contextualize_q_prompt | llm,
        get_session_history=lambda session_id: chat_history_storage.setdefault(
            session_id, ChatMessageHistory()
        ),
        input_messages_key="input",
        history_messages_key="history",
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def combined_chain(input_data):
        session_id = input_data.get("session_id")
        user_query = input_data.get("input")
        employee_id = input_data.get("employee_id")  # employee_id 추가

        # 히스토리 가져오기
        chat_history = chat_history_storage.get(session_id, ChatMessageHistory())
        history_as_list = chat_history.messages

        # 1. 사원 데이터 요약 생성 (히스토리에 추가하지 않음)
        employee_summary = ""
        employee_summary_document = None
        if employee_id:
            # DB에서 사원 데이터를 조회하고 요약
            employee_data = fetch_employee_data(employee_id)
            employee_summary_text = summarize_employee_data(employee_data, employee_id)
            employee_summary = generate_employee_summary_text(employee_summary_text)

            # 사원 요약 정보를 Document로 생성
            employee_summary_document = Document(
                page_content=f"질문한 사원(사용자)의 요약 정보:\n{employee_summary}",
                metadata={"source": "employee_summary"},
            )

        # 2. 질문 문맥화 처리
        contextualized_result = contextualize_chain.invoke(
            {"input": user_query, "history": history_as_list},
            {"configurable": {"session_id": session_id}},
        )

        # 문맥화된 질문이 비어 있을 경우 처리
        if not contextualized_result.content.strip():
            return {
                "contextualized_question": None,
                "retrieval_response": "구체적인 HR 관련 질문을 입력해 주세요.",
                "source_documents": [],
            }

        # 3. RAG 체인 실행
        retrieval_result = rag_chain.invoke(
            {
                "input": contextualized_result.content,
                "history": history_as_list,
            }
        )
        print(f"Retrieval Result: {retrieval_result}")
        print()

        # 4. RAG 결과와 사원 데이터 결합
        combined_context = retrieval_result.get("context", []) + history_as_list

        # 사원 요약 정보를 combined_context에 추가
        if employee_summary_document:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            split_documents = text_splitter.split_documents([employee_summary_document])
            combined_context.extend(split_documents)

        # 5. 최종 LLM 입력 구성
        llm_input = [
            SystemMessage(
                content="You are a helpful assistant. Use the following context and history to answer the user's question."
            ),
            HumanMessage(
                content=f"Context: {combined_context}\n\nQuestion: {contextualized_result.content}"
            ),
        ]

        # LLM 호출
        print(f"LLM input: {llm_input}")  # LLM Response 출력
        print()
        llm_response = llm.invoke(llm_input)
        print(f"LLM Response: {llm_response}")  # LLM Response 출력
        print()

        return {
            "contextualized_question": contextualized_result.content,
            "retrieval_response": llm_response.content,  # LLM 응답
            "source_documents": retrieval_result.get("context", []),
            "employee_data": employee_data,  # 사원 데이터 반환
        }

    return combined_chain


# 사원 데이터 초기화
employee_data = {}


@app.post("/query")
async def query(request: QueryRequest):

    # 실행 시간 측정
    start_time = time.time()

    global employee_data
    try:
        user_input = request.query
        session_id = request.session_id
        employee_id = request.employee_id  # employee_id 추가

        # "우리 프로그램 어때?"에 대한 특별한 응답 처리
        if user_input.strip() == "우리 프로그램 어때?":
            specific_answer = """ 이 프로그램은 인사, 근태, 휴가, 급여, 평가, 계약서 관리 등 **인사관리의 핵심 기능**을 모두 구현하며, 다양한 산업군에 적용 가능한 B2B 중심 솔루션을 개발했습니다.\n\n 24시간 AI 챗봇을 통해 단순 문의를 신속히 처리하고, 전자 계약서 관리와 업무 자동화를 통해 **작업 효율성을 대폭 향상**시켰습니다.\n\n저희는 이 프로그램을 단순한 도구가 아닌, 모든 회사의 더 나은 내일을 위한 든든한 동반자로 만들고 싶었습니다.\n\n작은 문제를 해결하며 쌓아온 시간들, 밤낮없이 고민한 노력들이 오늘 이 자리를 통해 여러분께 닿길 바랍니다.\n\n**끝으로, 긴 발표를 들어주신 심사위원님과 청취자 여러분께 진심으로 감사드립니다. \n저희의 열정과 노력에 따뜻한 관심과 응원을 부탁드립니다. 감사합니다!**"""

            # JSON 응답 데이터 생성
            response_content = {
                "contextualized_question": user_input,
                "answer": specific_answer,
                "selected_keyword": "Nothing",
                "source_documents": [],
                "history": [],
            }
            serialized_response = serialize_data(response_content)

            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f"실행 시간: {execution_time:.2f} seconds.")
            return JSONResponse(content=serialized_response)

        # 체인 생성 및 실행
        chain = create_chain_with_message_history()
        response = chain(
            {"input": user_input, "session_id": session_id, "employee_id": employee_id}
        )

        # 질문 및 답변 처리
        contextualized_question = response.get("contextualized_question", "").replace(
            "\\", ""
        )
        print("contextualized_question: ", contextualized_question)
        print()

        answer = response.get(
            "retrieval_response", "답변을 생성하지 못했습니다."
        ).replace("\\", "")

        serialized_documents = serialize_documents(response.get("source_documents", []))

        # 히스토리 가져오기
        chat_history = get_or_create_history(session_id)
        serialized_history = [
            {
                "type": message.type,  # 메시지 타입 (human/system 등)
                "content": message.content.replace("\n", " ").replace("\\", ""),
            }
            for message in chat_history.messages
        ]

        # 한국어 키워드와 매칭되는 딕셔너리
        keyword_dict = {
            "근퇴": "commute",
            "출퇴근": "commute",
            "재택": "remote",
            "초과 근무": "overtime",
            "초과근무": "overtime",
            "휴가": "vacation",
            "휴직": "leave",
            "복직": "leave",
            "출장": "business",
            "파견": "dispatch",
            "급여": "salary",
            "계약": "contract",
            "증명": "contract",
            "부서": "department",
            "평가": "evaluation",
        }

        # 정규표현식을 통해 키워드 검색
        if hasattr(contextualized_question, "content"):  # AIMessage인 경우 content 추출
            contextualized_result_text = contextualized_question.content
        else:  # 기본적으로 문자열로 처리
            contextualized_result_text = str(contextualized_question)

        pattern = re.compile("|".join(map(re.escape, keyword_dict.keys())))
        match = pattern.search(contextualized_result_text)

        # 매칭된 키워드로 값만 설정
        selected_keyword = keyword_dict[match.group()] if match else "Nothing"

        # 결과 출력
        print(f"선택된 딕셔너리: {selected_keyword}")

        # 응답 데이터 통합
        response_content = {
            "contextualized_question": contextualized_question,
            "answer": answer,
            "selected_keyword": selected_keyword,
            "source_documents": serialized_documents,
            "history": serialized_history,
        }
        # JSON 직렬화 보장
        serialized_response = serialize_data(response_content)

        end_time = time.time()
        # 실행 시간 로그 출력
        execution_time = end_time - start_time
        logging.info(f"실행 시간: {execution_time:.2f} seconds.")
        return JSONResponse(content=serialized_response)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in query: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))


def summarize_employee_data(employee_data, employee_id):
    """
    특정 사원의 데이터를 요약하여 간단한 형태로 변환.

    Args:
        employee_data (dict): 전체 사원 데이터.
        employee_id (int): 요약할 사원의 ID.

    Returns:
        dict: 요약된 데이터 사전 형태.
    """
    print("employee_data: ", employee_data)

    # 부서 구성원 디버깅
    for idx, member in enumerate(employee_data.get("department_members", [])):
        if "attendance_status_type_name" not in member:
            print(f"Missing attendance_status_type_name in record {idx}: {member}")

    summary = {
        "employee_id": employee_id,
        "employee_name": employee_data.get("employee", [{}])[0].get("name"),
        "role_info": employee_data.get("role", [{}])[0],
        "duty_info": employee_data.get("duty", [{}])[0],
        "position_info": employee_data.get("position", [{}])[0],
        "vacation_info": [],
        "department_info": employee_data.get("department", [{}])[0],
        "department_members": [
            {
                "member_name": member.get("name", "N/A"),
                "role": member.get("role_name", "N/A"),
                "email": member.get("email", "N/A"),
                "phone_number": member.get("phone_number", "N/A"),
                "attendance_status_type_name": member.get(
                    "attendance_status_type_name", "N/A"
                ),
                "manager_status": (
                    "Y" if member.get("manager_status", "N") == "Y" else "N"
                ),
            }
            for member in employee_data.get("department_members", [])
        ],
        "evaluation_info": [],
        "task_type_evaluation_info": [],
        "grades": [],
        "commute_records": employee_data.get("commute", []),
        "attendance_requests": [],
        "company_info": employee_data.get("company", [{}])[0],
        "evaluation_policies": [],
    }

    # 휴가 정보 요약
    vacations = employee_data.get("vacation", [])
    summary["vacation_info"] = [
        {
            "name": vacation.get("vacation_name", "이름 없음"),
            "remaining_days": vacation.get("vacation_left", 0),
            "used_days": vacation.get("vacation_used", 0),
        }
        for vacation in vacations
    ]

    # 평가 정책 정보 요약
    evaluation_policies = employee_data.get("evaluation_policy", [])
    summary["evaluation_policies"] = [
        {
            "year": policy.get("year", "N/A"),
            "half": policy.get("half", "N/A"),
            "start_date": policy.get("start_date", "N/A"),
            "end_date": policy.get("end_date", "N/A"),
        }
        for policy in evaluation_policies
    ]
    return summary


def fetch_employee_data(employee_id):
    """
    주어진 employee_id로 사원 관련 데이터를 조회합니다.

    Args:
        employee_id (int): 조회할 사원의 ID.

    Returns:
        dict: 사원 관련 데이터.
    """
    try:
        engine = connect_sqlalchemy()
        with engine.connect() as connection:
            queries = {
                "employee": "SELECT name, department_code, position_code, role_code, duty_code FROM employee WHERE employee_id = :employee_id",
                "vacation": """
                    SELECT vacation_name, vacation_left, vacation_used
                    FROM vacation
                    WHERE employee_id = :employee_id
                    AND vacation_left < 100
                """,
                "evaluation_policy": """
                    SELECT year, half, start_date, end_date
                    FROM evaluation_policy
                    ORDER BY year DESC, half DESC
                """,
                "evaluation": """
                    SELECT evaluation_type, fin_grade, fin_score, year, half
                    FROM evaluation
                    WHERE employee_id = :employee_id
                """,
                "task_type_evaluation": """
                    SELECT task_type_total_score, evaluation_policy_id, created_at
                    FROM task_type_eval
                    WHERE evaluation_id IN (
                        SELECT evaluation_id
                        FROM evaluation
                        WHERE employee_id = :employee_id
                    )
                """,
                "grade": """
                    SELECT grade_name, start_ratio, end_ratio, absolute_grade_ratio, evaluation_policy_id
                    FROM grade
                    WHERE evaluation_policy_id IN (
                        SELECT evaluation_policy_id
                        FROM task_type_eval
                        WHERE evaluation_id IN (
                            SELECT evaluation_id
                            FROM evaluation
                            WHERE employee_id = :employee_id
                        )
                    )
                """,
                "department": """
                    SELECT department_name FROM department WHERE department_code = (SELECT department_code FROM employee WHERE employee_id = :employee_id)
                """,
                "role": """
                    SELECT role_name FROM role WHERE role_code = (SELECT role_code FROM employee WHERE employee_id = :employee_id)
                """,
                "duty": """
                    SELECT duty_name FROM duty WHERE duty_code = (SELECT duty_code FROM employee WHERE employee_id = :employee_id)
                """,
                "position": """
                    SELECT position_name FROM position WHERE position_code = (SELECT position_code FROM employee WHERE employee_id = :employee_id)
                """,
                "department_members": """
                    WITH RECURSIVE SubDepartments AS (
                        SELECT department_code
                        FROM department
                        WHERE department_code = (SELECT department_code FROM employee WHERE employee_id = :employee_id)
                        
                        UNION ALL
                        
                        SELECT d.department_code
                        FROM department d
                        INNER JOIN SubDepartments sd ON d.upper_department_code = sd.department_code
                    )
                    SELECT dm.name, dm.role_name, dm.email, dm.phone_number, 
                           dm.attendance_status_type_name, dm.manager_status
                    FROM department_member dm
                    WHERE dm.department_code IN (SELECT department_code FROM SubDepartments);
                """,
                "attendance_requests": """
                    SELECT start_date, end_date, created_at, request_reason, request_status, canceled_at, cancel_reason
                    FROM attendance_request
                    WHERE employee_id = :employee_id
                """,
                "company": """
                    SELECT company_name, ceo, business_registration_number, company_address, company_phone_number, 
                           company_stamp_url, company_logo_url
                    FROM company
                    WHERE company_id = 1
                """,
            }

            # 데이터 가져오기
            data = {
                key: pd.read_sql(
                    text(query), connection, params={"employee_id": employee_id}
                ).to_dict(orient="records")
                for key, query in queries.items()
            }

        return data

    except Exception as e:
        print(f"Error fetching employee data: {e}")
        return {}


def generate_employee_summary_text(summary):
    """
    요약 데이터를 기반으로 각 데이터 유형별 문장을 생성.

    Args:
        summary (dict): 요약된 사원 데이터.

    Returns:
        str: 문장으로 구성된 요약 데이터.
    """
    employee_id = summary.get("employee_id", "알 수 없음")
    employee_name = summary.get("employee_name", "알 수 없음")
    role_info = summary.get("role_info", {}).get("role_name", "알 수 없음")
    duty_info = summary.get("duty_info", {}).get("duty_name", "알 수 없음")
    position_info = summary.get("position_info", {}).get("position_name", "알 수 없음")
    department_name = summary.get("department_info", {}).get(
        "department_name", "알 수 없음"
    )

    # 회사 정보
    company_info = summary.get("company_info", {})
    company_name = company_info.get("company_name", "알 수 없음")
    ceo = company_info.get("ceo", "알 수 없음")
    business_registration_number = company_info.get(
        "business_registration_number", "알 수 없음"
    )
    company_address = company_info.get("company_address", "알 수 없음")
    company_phone_number = company_info.get("company_phone_number", "알 수 없음")

    # 사원 기본 정보
    result = [
        f"질문사원 ID: {employee_id}",
        f"질문 사원 이름: {employee_name}",
        f"질문자 직위: {position_info}",
        f"질문자 직무: {duty_info}",
        f"질문자 직책: {role_info}",
        f"질문자 소속 부서: {department_name}",
        "\n질문자 소속 회사 정보:",
        f"- 회사명: {company_name}",
        f"- 대표자: {ceo}",
        f"- 사업자 등록번호: {business_registration_number}",
        f"- 주소: {company_address}",
        f"- 전화번호: {company_phone_number}",
    ]

    # 휴가 정보
    vacations = summary.get("vacation_info", [])
    if vacations:
        vacation_texts = [
            f"- {vacation['name']}: 남은 {vacation['remaining_days']}일, 사용 {vacation['used_days']}일"
            for vacation in vacations
        ]
        result.append(f"\n휴가 정보:\n" + "\n".join(vacation_texts))
    else:
        result.append("\n휴가 정보: 데이터가 없습니다.")

    # 부서 구성원
    department_members = summary.get("department_members", [])
    if department_members:
        member_texts = [
            f"- 이름: {member['member_name']}, 역할: {member['role']}, 이메일: {member['email']}, "
            f"전화번호: {member['phone_number']}, 출근 상태: {member['attendance_status_type_name']}, "
            f"부장 여부: {'부장' if member['manager_status'] == 'Y' else '부원'}"
            for member in department_members
        ]
        result.append(f"\n부서 구성원:\n" + "\n".join(member_texts))
    else:
        result.append("\n부서 구성원 정보: 데이터가 없습니다.")

    # 평가 정보
    evaluations = summary.get("evaluation_info", [])
    if evaluations:
        evaluation_texts = [
            f"- {evaluation['evaluation_type']} (년도: {evaluation['year']}, 반기: {evaluation['half']}): "
            f"등급: {evaluation['final_grade']}, 점수: {evaluation['final_score']}"
            for evaluation in evaluations
        ]
        result.append(f"\n평가 정보:\n" + "\n".join(evaluation_texts))
    else:
        result.append("\n평가 정보: 데이터가 없습니다.")

    # 평가 정책 정보 추가
    evaluation_policies = summary.get("evaluation_policies", [])
    if evaluation_policies:
        policy_texts = [
            f"- 연도: {policy['year']}, 반기: {policy['half']}, 시작일: {policy['start_date']}, 종료일: {policy['end_date']}"
            for policy in evaluation_policies
        ]
        result.append(f"\n평가 정책:\n" + "\n".join(policy_texts))
    else:
        result.append("\n평가 정책: 데이터가 없습니다.")

    # 근태 신청
    attendance_requests = summary.get("attendance_requests", [])
    if attendance_requests:
        attendance_texts = [
            f"- 신청일: {request['start_date']} ~ {request['end_date']}, 사유: {request['reason']}, 상태: {request['status']}"
            for request in attendance_requests
        ]
        result.append(f"\n근태 신청:\n" + "\n".join(attendance_texts))
    else:
        result.append("\n근태 신청: 데이터가 없습니다.")

    return "\n".join(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
