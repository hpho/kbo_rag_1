✅ 구성도

[User Query]
     ↓ (자연어)
[FastAPI 서버]
     ↓
[LLM (e.g. GPT-3.5)]
     ↓ (SQL 생성)
[SQLite/MySQL DB]
     ↓
[결과 JSON 응답]

🧱 기술 스택
LLM	OpenAI GPT-4o
DB	SQLite (or MySQL/PostgreSQL for 확장성)
자연어→ SQL 변환	LangChain SQLDatabaseChain
API 서버	FastAPI
SQLAlchemy (pandas)
streamlit

my_sql_api_project/
├── main.py                    ← FastAPI 앱
├── db/
│   └── kbo_stats.db           ← SQLite DB
├── .env                       ← API 키 등 민감정보
├── requirements.txt
