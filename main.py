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

    # 질문 문맥화 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문 답변 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. Please use imogi with the answer. \
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
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
        model="gpt-4",
        temperature=0.2,
    )

    # RunnableWithMessageHistory: 대화 히스토리 관리
    contextualize_chain = RunnableWithMessageHistory(
        runnable=contextualize_q_prompt | llm,
        get_session_history=lambda session_id: chat_history_storage.setdefault(
            session_id, ChatMessageHistory()
        ),
        input_messages_key="input",
        history_messages_key="history",
    )

    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # 체인을 결합
    def combined_chain(input_data):
        session_id = input_data.get("session_id")
        user_query = input_data.get("input")

        # 1. 질문 문맥화 처리
        contextualized_result = contextualize_chain.invoke(
            {"input": user_query, "history": chat_history_storage.get(session_id, [])},
            {"configurable": {"session_id": session_id}},
        )
        print(f"Contextualized Question: {contextualized_result}")

        # 2. QA 체인 실행
        retrieval_result = retrieval_qa_chain.invoke(
            {"query": contextualized_result.content, "history": chat_history_storage.get(session_id, [])}
        )
        print(f"Retrieval Result: {retrieval_result}")

        return {
            "contextualized_question": contextualized_result.content,
            "retrieval_response": retrieval_result.get("result"),
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
        # 요청 데이터
        user_input = request.query
        session_id = request.session_id

        # 체인 생성
        chain_with_message_history = create_chain_with_message_history()

        # 체인을 통해 질문 전달 및 응답 처리
        response = chain_with_message_history(
            {"input": user_input, "session_id": session_id}
        )

        # 문서 직렬화
        serialized_documents = serialize_documents(response["source_documents"])

        # 응답 데이터 구조
        return JSONResponse(
            content={
                "contextualized_question": response["contextualized_question"],  # 문맥화된 질문
                "answer": response["retrieval_response"],  # 최종 응답
                "source_documents": serialized_documents,  # 문서 정보
            }
        )

    except Exception as e:
        # 에러 핸들링 및 로그 출력
        error_trace = traceback.format_exc()
        print(f"Error in query: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
