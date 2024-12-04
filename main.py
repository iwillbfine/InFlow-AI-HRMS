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

import traceback  # 에러 디버깅 및 트레이스백 추적
import uvicorn  # FastAPI 서버 실행


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
        documents = []
        for file in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load_and_split())
            elif file.endswith(".sql"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load_and_split())

        split_docs = text_splitter.split_documents(documents)
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


# 입력 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str
    session_id: str


# 답변 문서 serializing
def serialize_documents(documents):
    serialized_docs = []
    for doc in documents:
        serialized_docs.append(
            {
                "page_content": doc.page_content.replace("\\", ""),
                "metadata": doc.metadata,
            }
        )
    return serialized_docs


# 히스토리 생성 및 조회 

chat_history_storage = {} # 대화 히스토리 저장소

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
    You are an HR management system chatbot capable of answering questions related to salary, leave, attendance, and performance evaluation.
    Leverage the retrieved context and chat history to provide the most accurate and helpful answer.

    If the context or history does not provide enough information, respond as follows:
    - If you can guess the intent, provide a relevant response or a suggestion.
    - If no answer can be reasonably inferred, reply politely: "질문에 대해 정확한 답변을 드리기 어려워요. 조금 더 구체적으로 질문해 주시면 감사하겠습니다."

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
        combined_context = retrieval_result.get("context", []) + history_as_list

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


@app.post("/query")
async def query(request: QueryRequest):
    try:
        user_input = request.query
        session_id = request.session_id

        # 체인 생성 및 실행
        chain = create_chain_with_message_history()
        response = chain({"input": user_input, "session_id": session_id})

        # 이스케이프 문자 제거
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
