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

import traceback  # 에러 디버깅 및 트레이스백 추적
import uvicorn  # FastAPI 서버 실행


# .env 파일 로드
load_dotenv()

# 데이터 경로 설정
DATA_DIR = r"C:\lecture\FinalProject\InFlow-AI\data"
CHROMA_DB_DIR = r"C:\lecture\FinalProject\InFlow-AI\chromadb"
CSV_DIR = r"C:\lecture\FinalProject\InFlow-AI\chromadb\csv_files"

# MariaDB 연결 정보
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "test_db")

# OpenAI API 키 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# OpenAI 임베딩 설정
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 벡터 스토어 초기화
vectorstore = None


# CSV 디렉토리와 Chroma DB 디렉토리를 초기화 및 생성
def initialize_directories():
    # 디렉토리 초기화
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    if os.path.exists(CSV_DIR):
        shutil.rmtree(CSV_DIR)
    os.makedirs(CSV_DIR, exist_ok=True)


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


# 모든 테이블을 병렬로 CSV로 저장
def export_tables_to_csv():

    # 디렉토리 유효성 검사
    if not os.path.exists(CSV_DIR):
        raise FileNotFoundError(f"CSV directory does not exist: {CSV_DIR}")

    engine = connect_sqlalchemy()  # SQLAlchemy 엔진 사용
    try:
        # 모든 테이블 이름 가져오기
        with engine.connect() as connection:
            # 원시 SQL 실행
            result = connection.execute(text("SHOW TABLES;"))
            tables = [row[0] for row in result]

        # 병렬 처리로 각 테이블 데이터를 CSV로 저장
        with ThreadPoolExecutor() as executor:
            executor.map(
                lambda table: export_table_to_csv_parallel(table, engine), tables
            )

    except Exception as e:
        print(f"Error exporting all tables: {e}")


# 특정 테이블을 병렬로 CSV로 저장
def export_table_to_csv_parallel(table_name, engine):
    try:
        # 디렉토리 유효성 검사
        if not os.path.exists(CSV_DIR):
            raise FileNotFoundError(f"CSV directory does not exist: {CSV_DIR}")

        # employee 테이블일 경우 민감 정보 제외
        if table_name == "employee":
            query = """
            SELECT 
                employee_id,
                employee_number,
                employee_role,
                gender,
                name,
                birth_date,
                email,
                phone_number,
                profile_img_url,
                join_date,
                resignation_date,
                resignation_status,
                department_code,
                attendance_status_type_code,
                position_code,
                role_code,
                duty_code
            FROM employee
            """
        else:
            # 다른 테이블은 전체 데이터를 가져옴
            query = f"SELECT * FROM {table_name}"

        # 데이터 프레임으로 가져오기
        df = pd.read_sql(query, engine)

        # CSV 파일로 저장
        csv_path = os.path.join(CSV_DIR, f"{table_name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Exported {table_name} to {csv_path}")

    except Exception as e:
        print(f"Error exporting table {table_name}: {e}")


# CSV.pdf 데이터를 벡터화하고 Chroma DB에 저장
def store_files_in_chroma():
    """CSV와 PDF 파일을 벡터 스토어에 저장"""
    global vectorstore
    documents = []

    # CSV 파일 처리
    for csv_file in os.listdir(CSV_DIR):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(CSV_DIR, csv_file)
            df = pd.read_csv(csv_path)

            if df.empty:
                print(f"Skipped empty file: {csv_file}")
                continue

            for _, row in df.iterrows():
                content = " ".join([f"{col}:{val} " for col, val in row.items()])
                documents.append(
                    Document(page_content=content, metadata={"table": csv_file})
                )

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
            export_tables_to_csv()  # 2. 모든 테이블 데이터를 CSV로 저장
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
        print("Initializing directories...")
        initialize_directories()

        print("Exporting tables to CSV...")
        export_tables_to_csv()

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
    try:
        # 애플리케이션 초기화 시 실행할 작업
        print("Application starting...")
        initialize_directories()  # 디렉토리 초기화
        export_tables_to_csv()  # CSV 생성
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


# 히스토리 생성 및 조회

chat_history_storage = {}  # 대화 히스토리 저장소


def get_or_create_history(session_id):
    if session_id not in chat_history_storage:
        chat_history_storage[session_id] = ChatMessageHistory()
    return chat_history_storage[session_id]


# 체인 생성 함수
def create_chain_with_message_history():
    global rag_chain  # 전역 변수를 사용하도록 선언

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 검색 결과 제한

    # 질문 문맥화 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. 질문을 생성하세요."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문 답변 프롬프트
    qa_system_prompt = """
    You are an HR management system chatbot. Use the retrieved context, including the document metadata (table names and descriptions), to generate an accurate and helpful answer.\

    For each table referenced in the retrieved context, include its Korean name and description in the response if relevant.\

    If the context or history does not provide enough information, respond as follows:\
    - If you can guess the intent, provide a relevant response or a suggestion.\
    - If no answer can be reasonably inferred, reply politely: "질문에 대해 정확한 답변을 드리기 어려워요. 조금 더 구체적으로 질문해 주시면 감사하겠습니다."\

    대답은 반드시 한국어로 작성하고, 반드시 존댓말을 써주세요.\

    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # LLM 설정
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.2,
    )

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

    # 체인결합
    def combined_chain(input_data):
        session_id = input_data.get("session_id")
        user_query = input_data.get("input")

        # 히스토리 가져오기
        chat_history = chat_history_storage.get(session_id, ChatMessageHistory())
        history_as_list = chat_history.messages

        # 1. 질문 문맥화 처리
        contextualized_result = contextualize_chain.invoke(
            {"input": user_query, "history": history_as_list},
            {"configurable": {"session_id": session_id}},
        )
        print(f"Contextualized Question: {contextualized_result}")
        print()

        # 문맥화된 질문이 비어 있을 경우 처리
        if not contextualized_result.content.strip():
            return {
                "contextualized_question": None,
                "retrieval_response": "구체적인 HR 관련 질문을 입력해 주세요.",
                "source_documents": [],
            }

        # 2. RAG 체인 실행
        retrieval_result = rag_chain.invoke(
            {
                "input": contextualized_result.content,
                "history": history_as_list,
            }
        )
        print(f"Retrieval Result: {retrieval_result}")
        print()

        # 검색 결과와 히스토리를 결합하여 LLM에게 더 풍부한 컨텍스트 제공
        combined_context = (
            annotate_with_table_names(retrieval_result.get("context", []))
            + history_as_list
        )

        # 3. 최종 응답 생성
        llm_input = [
            SystemMessage(
                content="You are a helpful assistant. Use the following context and history to answer the user's question."
            ),
            HumanMessage(
                content=f"Context: {combined_context}\n\nQuestion: {contextualized_result.content}"
            ),
        ]

        # LLM 호출 (invoke 사용)
        llm_response = llm.invoke(llm_input)
        print(f"LLM Response: {llm_response}")  # LLM Response 출력
        print()

        return {
            "contextualized_question": contextualized_result.content,
            "retrieval_response": llm_response.content,  # LLM 응답
            "source_documents": retrieval_result.get("context", []),
        }

    return combined_chain


table_name_mapping = {
    "monthly_employee_num_statistics": "월별 사원 수 통계 테이블 (사원의 월별 변화 추이를 포함)",
    "monthly_department_overtime_allowance_statistics": "월별 부서별 초과근무 수당 통계 테이블",
    "semiannual_department_performance_ratio_statistics": "반기별 부서 성과 비율 통계 테이블",
    "feedback": "피드백 테이블 (사원 및 부서에 대한 피드백 정보를 포함)",
    "task_eval": "과제 평가 테이블 (개별 과제에 대한 평가 기록 포함)",
    "task_type_eval": "과제 유형 평가 테이블 (각 유형별 평가 정보 포함)",
    "grade": "등급 테이블 (사원의 평가 등급 정보를 포함)",
    "evaluation": "평가 테이블 (사원의 성과 및 능력 평가 기록 포함)",
    "task_item": "과제 항목 테이블 (세부 과제 항목 정보 포함)",
    "evaluation_policy": "평가 정책 테이블 (평가 기준 및 정책 정보 포함)",
    "task_type": "과제 유형 테이블 (과제의 분류 및 유형 정보 포함)",
    "business_trip": "출장 테이블 (사원의 출장 기록 및 세부 정보 포함)",
    "leave_return": "복직 테이블 (휴직 후 복직 정보 포함)",
    "commute": "사원별 출퇴근 테이블이력 (사원의 출퇴근 시간 및 기록 포함)",
    "attendance_request_file": "근태 요청 파일 테이블 (근태 요청에 첨부된 파일 정보 포함)",
    "attendance_request": "근태 요청 테이블 (사원의 근태 변경 요청 기록 포함)",
    "attendance_request_type": "근태 요청 유형 테이블 (근태 요청의 분류 및 유형 정보 포함)",
    "payment": "지급 테이블 (사원의 급여 및 보너스 지급 내역 포함)",
    "irregular_allowance": "비정기 수당 테이블 (특별 수당 및 비정기 수당 정보 포함)",
    "public_holiday": "공휴일 테이블 (회사에서 인정하는 공휴일 정보 포함)",
    "tax_credit": "세액 공제 테이블 (사원의 세액 공제 정보 포함)",
    "non_taxable": "비과세 항목 테이블 (비과세 수당 및 기타 항목 정보 포함)",
    "major_insurance": "주요 보험 테이블 (4대 보험 관련 정보 포함)",
    "earned_income_tax": "근로 소득세 테이블 (사원의 소득세 계산 정보 포함)",
    "annual_vacation_promotion_policy": "연차 촉진 정책 테이블 (연차 사용 및 촉진 정책 정보 포함)",
    "vacation_request_file": "휴가 요청 파일 테이블 (휴가 요청에 첨부된 파일 정보 포함)",
    "vacation_request": "휴가 요청 테이블 (사원의 휴가 신청 기록 포함)",
    "vacation": "사원별 지급 받은 휴가 테이블 ",
    "vacation_policy": "휴가 정책 테이블 (연차 및 유급/무급 휴가 정책 관련 정보 포함)",
    "vacation_type": "휴가 유형 테이블 (연차, 병가 등 휴가 유형 정보 포함)",
    "department_member": "부서 구성원 테이블 (부서별 소속 사원 정보 포함)",
    "appointment": "인사 발령 테이블 (사원의 인사 발령 기록 포함)",
    "appointment_item": "인사 발령 항목 테이블 (세부 인사 발령 항목 정보 포함)",
    "discipline_reward": "징계 및 포상 테이블 (사원의 징계 및 포상 기록 포함)",
    "language_test": "어학 시험 테이블 (사원의 어학 시험 기록 및 점수 포함)",
    "language": "언어정보 테이블 (지원 가능한 언어 정보 포함)",
    "qualification": "자격증 테이블 (사원의 자격증 정보 포함)",
    "contract": "계약 테이블 (사원의 계약 정보 및 내용 포함)",
    "career": "경력 테이블 (사원의 이전 경력 및 기록 포함)",
    "education": "학력 테이블 (사원의 학력 정보 포함)",
    "family_member": "가족 구성원 테이블 (사원의 부양 가족 정보 포함)",
    "family_relationship": "가족 관계 테이블 (가족 구성원의 관계 정보 포함)",
    "employee": "사원 정보 테이블 (사원의 개인 정보와 직책 정보를 포함)",
    "duty": "직무 테이블 (사원의 직무 및 역할 정보 포함)",
    "role": "역할 테이블 (사원의 역할 및 권한 정보 포함)",
    "position": "직위 테이블 (사원의 직위 정보 포함)",
    "attendance_status_type": "근태 상태 유형 테이블 (출근, 조퇴 등 근태 상태 정보 포함)",
    "department": "부서 테이블 (부서의 이름 및 코드 정보 포함)",
    "company": "회사 테이블 (회사 정보 및 조직 구조 포함)",
}


#  테이블명을 한국어 설명으로 매핑
def map_table_name_to_korean(table_name):
    return table_name_mapping.get(table_name, f"알 수 없는 테이블: {table_name}")


# 문서에 한국어 테이블명 주석 추가
def annotate_with_table_names(documents):

    for doc in documents:
        table_name = doc.metadata.get("table")
        if table_name:
            doc.metadata["table_korean"] = map_table_name_to_korean(table_name)
    return documents


@app.post("/query")
async def query(request: QueryRequest):
    try:
        user_input = request.query
        session_id = request.session_id

        # 체인 생성 및 실행
        chain = create_chain_with_message_history()
        response = chain({"input": user_input, "session_id": session_id})

        # 검색 결과 가공
        source_documents = annotate_with_table_names(
            response.get("source_documents", [])
        )

        contextualized_question = response.get("contextualized_question", "").replace(
            "\\", ""
        )

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

        return JSONResponse(
            content={
                "contextualized_question": contextualized_question,
                "answer": answer,
                "source_documents": serialized_documents,
                "history": serialized_history,  # 대화 히스토리 추가
            }
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in query: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
