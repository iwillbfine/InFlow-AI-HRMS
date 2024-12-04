import os
import pymysql
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma  # 벡터 스토어로 Chroma를 사용
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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


# MariaDB 연결
def connect_mariadb():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
        )
        return conn
    except Exception as e:
        print(f"Error connecting to MariaDB: {e}")
        raise


# 데이터베이스 테이블 데이터를 CSV로 저장
def export_tables_to_csv():
    os.makedirs(CSV_DIR, exist_ok=True)
    conn = connect_mariadb()
    cursor = conn.cursor()

    try:
        # 모든 테이블 이름 가져오기
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            csv_path = os.path.join(CSV_DIR, f"{table_name}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Exported {table_name} to {csv_path}")

    except Exception as e:
        print(f"Error exporting tables: {e}")
    finally:
        conn.close()


# CSV 데이터를 벡터화하고 Chroma DB에 저장
def store_csv_in_chroma():
    global vectorstore  # 글로벌 변수 사용
    documents = []

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
        query = "윤지혜 사원의 잔여 휴가 일수는?"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 검색 결과 제한
        results = retriever.get_relevant_documents(query)

        print("Query:", query)
        print("Results:")
        for result in results:
            print(result.page_content)
            print("Metadata:", result.metadata)
    except Exception as e:
        print(f"Error testing RAG query: {e}")


# 통합 실행
if __name__ == "__main__":
    # 1. 모든 테이블 데이터를 CSV로 저장
    export_tables_to_csv()

    # 2. CSV 데이터를 벡터 DB에 저장
    store_csv_in_chroma()

    # 3. RAG 시스템 테스트
    test_rag_query()
