import os
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.sql import text  # SQLAlchemy의 text 모듈 추가
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma  # 벡터 스토어로 Chroma를 사용
from langchain_openai import OpenAIEmbeddings  # 업데이트된 import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor  # 병렬 처리를 위한 모듈
from datetime import datetime  # 배치 주기를 위한 모듈
import time  # 추가: 시간 대기를 위한 모듈
import shutil  # 디렉토리 삭제를 위한 모듈

# .env 파일 로드
load_dotenv()

# MariaDB 연결 정보
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "test_db")

# 경로 설정
CSV_DIR = r"C:\lecture\FinalProject\InFlow-AI\db_test\csv_files"
CHROMA_DB_DIR = r"C:\lecture\FinalProject\InFlow-AI\db_test"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 설정
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 글로벌 변수로 선언
vectorstore = None


# CSV 디렉토리와 Chroma DB 디렉토리를 초기화 및 생성
def initialize_directories():
    # Chroma DB 디렉토리 초기화
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)  # 기존 Chroma DB 디렉토리 삭제
        print(f"Cleared Chroma DB directory: {CHROMA_DB_DIR}")
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)  # Chroma DB 디렉토리 재생성
    print(f"Created Chroma DB directory: {CHROMA_DB_DIR}")

    # CSV 디렉토리 초기화
    if os.path.exists(CSV_DIR):
        shutil.rmtree(CSV_DIR)  # 기존 CSV 디렉토리 삭제
        print(f"Cleared CSV directory: {CSV_DIR}")
    os.makedirs(CSV_DIR, exist_ok=True)  # CSV 디렉토리 재생성
    print(f"Created CSV directory: {CSV_DIR}")


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


# CSV 데이터를 벡터화하고 Chroma DB에 저장
def store_csv_in_chroma():
    global vectorstore  # 글로벌 변수 사용
    documents = []

    # csv 파일 변환
    try:
        for csv_file in os.listdir(CSV_DIR):
            if csv_file.endswith(".csv"):
                csv_path = os.path.join(CSV_DIR, csv_file)
                df = pd.read_csv(csv_path)

                # 데이터가 비어 있는 경우 스킵
                if df.empty:
                    print(f"Skipped empty file: {csv_file}")
                    continue

                # 데이터 프레임의 텍스트 데이터를 Document로 변환
                for _, row in df.iterrows():
                    content = " ".join([f"{col}: {val}" for col, val in row.items()])
                    documents.append(
                        Document(page_content=content, metadata={"table": csv_file})
                    )

        # 문서 분할 및 벡터화
        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            raise ValueError("No documents to store in Chroma.")

        # Chroma DB에 저장
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,  # 임베딩 함수 전달
            persist_directory=CHROMA_DB_DIR,
        )
        print("Data stored in Chroma DB.")

    except Exception as e:
        print(f"Error storing data in Chroma: {e}")


# RAG 테스트
def test_rag_query():
    global vectorstore  # 글로벌 변수 사용
    try:
        # 질문 테스트
        query = "윤지혜 사원의 정보는?"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # 검색 결과 제한
        results = retriever.invoke(query)

        print("Query:", query)
        print("Results:")
        for result in results:
            print(result.page_content)
            print("Document content:", result.page_content)  # 벡터화된 데이터 확인
            print("Metadata:", result.metadata)
    except Exception as e:
        print(f"Error testing RAG query: {e}")


# 주기적 배치 실행
def run_batch():
    while True:
        print(f"Batch started at {datetime.now()}")

        # 디렉토리 초기화 및 생성
        initialize_directories()

        # 유효성 검사 (추가)
        if not os.path.exists(CSV_DIR):
            print("CSV directory is missing. Aborting batch.")
            return

        # 1. 모든 테이블 데이터를 CSV로 저장
        export_tables_to_csv()

        # 2. CSV 데이터를 벡터 DB에 저장
        store_csv_in_chroma()

        print(f"Batch completed at {datetime.now()}")

        # 3. RAG 시스템 테스트
        test_rag_query()

        # 4. 1일 대기
        print("Waiting for 24 hours...")
        time.sleep(24 * 60 * 60)  # 1일(24시간)


# 통합 실행
if __name__ == "__main__":
    # 배치 실행
    run_batch()
