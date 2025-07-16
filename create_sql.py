import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import AzureChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain

# Load .env
load_dotenv()

# FastAPI app
app = FastAPI()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
end_point = os.getenv("AZURE_OPENAI_END_POINT")
# 2. SQLite 연결
engine = create_engine("sqlite:///kbo_stats.db")
db = SQLDatabase(engine)

# 3. LLM 연결
from langchain.chat_models import AzureChatOpenAI
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=end_point,
    api_key=api_key,
    api_version="2024-03-01-preview"
)

# 4. SQL Chain
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
class QueryRequest(BaseModel):
    question: str

# API endpoint
@app.post("/query")
async def query_db(req: QueryRequest):
    try:
        result = db_chain.run(req.question)
        return {"question": req.question, "result": result}
    except Exception as e:
        return {"error": str(e)}