import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel


# LangChain 관련 모듈

# vectorstore 관련
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Document 객체 가져오기
from langchain_community.vectorstores import Chroma


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser


# 에러 문구 트레이스
import traceback

import uvicorn

# .env 파일 로드
load_dotenv()


# 데이터 경로 설정
DATA_DIR = r"C:\lecture\FinalProject\InFlow-AI\data"
CHROMA_DB_DIR = r"C:\lecture\FinalProject\InFlow-AI\chromadb"

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore

    try:
        # PDF와 SQL 파일 로드
        documents = []
        for file in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load_and_split())
            elif file.endswith(".sql"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load_and_split())

        # 문서 청크 분할
        split_docs = text_splitter.split_documents(documents)

        # Chroma 벡터 스토어 생성 및 저장
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
        print("Chroma Vectorstore Initialized.")

    except Exception as e:
        print(f"Error initializing vectorstore: {e}")

    yield


# FastAPI 인스턴스 생성
app = FastAPI(lifespan=lifespan)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 대화 히스토리 저장소
chat_history_storage = {}

# 전역 변수로 RAG 체인을 관리
rag_chain = None


# 체인 생성 함수
def create_chain_with_message_history():
    global rag_chain  # 전역 변수를 사용하도록 선언

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 질문에 정확하고 관련된 답변을 제공하는 AI 도우미입니다.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # LLM 설정
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4",
        temperature=0.2,
    )

    # RunnableWithMessageHistory: 대화 히스토리 관리
    message_chain = RunnableWithMessageHistory(
        runnable=prompt | llm,
        get_session_history=lambda session_id: chat_history_storage.setdefault(
            session_id, ChatMessageHistory()
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Retrieval QA 체인 생성
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # 체인을 분리하고 수동으로 연결
    def combined_chain(input_data):
        # 메시지 체인 실행
        session_id = input_data.get("session_id")  # session_id를 가져옵니다.
        history_result = message_chain.invoke(
            {"input": input_data["input"]}, {"configurable": {"session_id": session_id}}
        )
        print(f"History Result: {history_result}")  # 디버깅용 로그

        # AIMessage 객체의 content를 추출
        history_content = history_result.content
        print(f"History Content: {history_content}")  # 디버깅용 로그

        # 히스토리에서 생성된 텍스트를 retriever로 전달
        retrieval_result = retrieval_qa_chain.invoke({"query": history_content})
        print(f"Retrieval Result: {retrieval_result}")  # 디버깅용 로그

        # `result`와 `source_documents`를 반환
        return {
            "result": retrieval_result.get("result"),
            "source_documents": retrieval_result.get("source_documents", []),
        }

    return combined_chain


# 입력 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str  # 사용자가 보낼 질문
    session_id: str  # 세션 ID


def serialize_documents(documents):
    """Document 객체 리스트를 JSON 직렬화가 가능한 딕셔너리 리스트로 변환"""
    serialized_docs = []
    for doc in documents:
        serialized_docs.append(
            {
                "page_content": doc.page_content,  # 문서 내용
                "metadata": doc.metadata,  # 메타데이터
            }
        )
    return serialized_docs


@app.post("/query")
async def query(request: QueryRequest):
    try:
        # 입력 데이터 접근
        user_input = request.query
        session_id = request.session_id

        # 입력 데이터를 문자열로 강제 변환
        if not isinstance(user_input, str):
            user_input = str(user_input)

        # RunnableWithMessageHistory로 자동 히스토리 관리 설정
        chain_with_message_history = create_chain_with_message_history()

        # 질문 전달 및 응답 받기
        print(
            f"Invoking chain with input: {user_input} and session_id: {session_id}"
        )  # 디버깅 로그 추가
        response = chain_with_message_history(
            {"input": user_input, "session_id": session_id}
        )
        print(f"Response from chain: {response}")  # 응답 디버깅

        # `source_documents`를 직렬화 가능한 형태로 변환
        serialized_documents = serialize_documents(response["source_documents"])

        # 응답 반환
        return JSONResponse(
            content={
                "answer": response["result"],  # 응답 텍스트
                "source_documents": serialized_documents,  # 직렬화된 문서
            }
        )

    except Exception as e:
        # 에러 로그 출력
        error_trace = traceback.format_exc()
        print(f"Error in query: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
