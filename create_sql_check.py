from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import AzureChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain

# Load .env
load_dotenv()

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

# 5. 실행
question = "2024년 홈런 10개 이상 기록한 LG 타자는 누구야? sql table은 batter야 그리고 쿼리를 마크다운 없이 출력해주세요."
result = db_chain.run(question)
print(result)
if result[0][0] == "오스틴":
    print(True)