
"""
Airline Customer Support – FastAPI Backend
==========================================
Setup in GitHub Codespaces
  pip install fastapi uvicorn psycopg2-binary \
              langchain langchain-core langchain-community langchain-openai \
              langchain-pinecone langchain-text-splitters pymupdf pinecone-client \
              langgraph openai

  export GROQ_API_KEY=<your-key>
  export PINECONE_API_KEY=<your-key>
  export POSTGRES_USER=<user>
  export SQLPWD=<password>

  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import psycopg2
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Airline Customer Support API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
OPEN_API_KEY     = os.getenv("OPEN_API_KEY", "")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX    = "airline-faq-index"
db_params = {
    "host":     os.getenv("DB_HOST", "aws-1-ap-northeast-2.pooler.supabase.com"),
    "port":     os.getenv("DB_PORT", "5432"),
    "user":     os.getenv("POSTGRES_USER", ""),
    "password": os.getenv("SQLPWD", ""),
    "dbname":   "postgres",
}

# ── LLM ────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="openai/gpt-oss-120b", temperature=0,
    api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
)

# ── Helpers ────────────────────────────────────────────────────────────────
def execute_sql_query(query: str) -> list:
    conn = psycopg2.connect(**db_params)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

@tool
def execute_flight_query(sql_query: str) -> str:
    """Execute a SELECT SQL query against the airline flights database."""
    try:
        rows = execute_sql_query(sql_query)
        return str(rows) if rows else "No flights found."
    except Exception as e:
        return f"Database error: {e}"

def format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)

# ── Chains / Agent (initialised once at startup) ───────────────────────────
def build_chains():
    # Classifier
    clf_prompt = ChatPromptTemplate.from_template(
        "Classify as Need SQL / Non SQL / Out of Context. Reply with the label only.\n{query}"
    )
    classifier = clf_prompt | llm | StrOutputParser()

    # SQL generation
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a single SELECT SQL for the flights table. No markdown, no explanation."),
        ("human", "Query: {query}"),
    ])
    sql_chain = sql_prompt | llm | StrOutputParser()

    # SQL agent
    agent = create_react_agent(llm, tools=[execute_flight_query])

    # RAG
    pc = Pinecone(api_key=PINECONE_API_KEY)
    vs = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY this context:\n{context}"),
        ("human", "{question}"),
    ])
    rag = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser())

    # Fallback
    fb_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an airline support bot. Politely explain you only handle airline topics."),
        ("human", "{query}"),
    ])
    fallback = fb_prompt | llm | StrOutputParser()

    # Input guardrail
    ig_prompt = ChatPromptTemplate.from_template(
        "Is this query safe? Reply SAFE or UNSAFE: <reason>.\nQuery: {query}"
    )
    in_guard = ig_prompt | llm | StrOutputParser()

    # Output guardrail
    og_prompt = ChatPromptTemplate.from_template(
        "Is this response safe to show? Reply SAFE or UNSAFE: <reason>.\nResponse: {response}"
    )
    out_guard = og_prompt | llm | StrOutputParser()

    return classifier, sql_chain, agent, rag, fallback, in_guard, out_guard


classifier_chain, sql_gen_chain, sql_agent, rag_chain, fallback_chain, in_guard_chain, out_guard_chain = build_chains()

BLOCKED_IN  = "I cannot process that request – it appears to violate safety guidelines."
BLOCKED_OUT = "I encountered an issue preparing the response. Please contact support."


def process(query: str) -> dict:
    # Input guardrail
    v = in_guard_chain.invoke({"query": query}).strip()
    if not v.upper().startswith("SAFE"):
        return {"category": "BLOCKED", "sql": None, "response": BLOCKED_IN}

    category = classifier_chain.invoke({"query": query}).strip()
    sql_used = None

    if category == "Need SQL":
        sql_used = sql_gen_chain.invoke({"query": query})
        msg = f"Customer question: {query}\n\nSQL: {sql_used}\n\nAnswer clearly."
        result = sql_agent.invoke({"messages": [{"role": "user", "content": msg}]})
        raw = result["messages"][-1].content
    elif category == "Non SQL":
        raw = rag_chain.invoke(query)
    else:
        raw = fallback_chain.invoke({"query": query})

    # Output guardrail
    ov = out_guard_chain.invoke({"response": raw}).strip()
    final = raw if ov.upper().startswith("SAFE") else BLOCKED_OUT
    return {"category": category, "sql": sql_used, "response": final}


# ── Request / Response models ───────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query:    str
    category: Optional[str]
    sql:      Optional[str]
    response: str

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Airline Customer Support API – POST /query to get started."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = process(req.query)
    return {"query": req.query, **result}
