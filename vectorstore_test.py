import os
import re
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# 데이터 경로 설정
SQL_FILE_PATH = r"C:\lecture\FinalProject\InFlow-AI\data\inflow_dml.sql"
CHROMA_DB_DIR = r"C:\lecture\FinalProject\InFlow-AI\chromadb_test"

# OpenAI 임베딩 설정
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)


def clear_vectorstore(directory):
    """기존 벡터스토어 삭제"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Cleared existing vectorstore at {directory}.")
    else:
        print(f"No existing vectorstore found at {directory}.")


def parse_sql_data(file_path):
    """SQL 파일에서 유의미한 데이터를 추출"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sql_content = f.read()

        # INSERT 구문에서 데이터를 추출
        matches = re.findall(
            r"INSERT INTO employee.*?VALUES\s*\((.*?)\);", sql_content, re.DOTALL
        )
        extracted_data = []
        for match in matches:
            rows = match.split("),")
            for row in rows:
                row_clean = row.replace("\n", "").strip().strip("()")
                extracted_data.append(
                    Document(page_content=row_clean, metadata={"source": file_path})
                )

        print(f"Extracted {len(extracted_data)} rows from SQL data.")
        return extracted_data

    except Exception as e:
        print(f"Error parsing SQL file: {e}")
        return []


def embed_extracted_data(data, directory):
    """추출된 데이터를 벡터스토어에 임베딩"""
    try:
        vectorstore = Chroma.from_documents(
            data, embeddings, persist_directory=directory
        )
        vectorstore.persist()
        print("Documents successfully embedded into the vectorstore.")
        return vectorstore
    except Exception as e:
        print(f"Error embedding documents: {e}")
        return None


def test_similarity_search(vectorstore, query, k=5):
    """벡터스토어에서 특정 키워드 검색"""
    try:
        results = vectorstore.similarity_search(query, k=k)
        print(f"Found {len(results)} matching documents for query: {query}")
        for idx, result in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"Content: {result.page_content}")
            print(f"Metadata: {result.metadata}")
    except Exception as e:
        print(f"Error during similarity search: {e}")


if __name__ == "__main__":
    # 기존 벡터스토어 삭제
    clear_vectorstore(CHROMA_DB_DIR)

    # SQL 데이터 파싱
    extracted_data = parse_sql_data(SQL_FILE_PATH)
    if not extracted_data:
        print("No data extracted. Exiting...")
        exit()

    # 벡터스토어 생성 및 임베딩
    vectorstore = embed_extracted_data(extracted_data, CHROMA_DB_DIR)
    if not vectorstore:
        print("Vectorstore creation failed. Exiting...")
        exit()

    # 검색 테스트
    print("\n=== Testing Similarity Search ===")
    test_query = "윤지혜"
    test_similarity_search(vectorstore, test_query)
