import pandas as pd
from sqlalchemy import create_engine

# SQLite DB 연결 (없으면 생성됨)
engine = create_engine("sqlite:///kbo_stats.db")

excel_path = r"kbo_statiz.xlsx"
xls = pd.ExcelFile(excel_path)
for i in xls.sheet_names:
    # CSV 파일 로드
    df = pd.read_excel("kbo_statiz.xlsx", sheet_name=i)
    if "2024" in i:
        df['YEAR'] = 2024
    elif "2025" in i:
        df['YEAR'] = 2025

    # 테이블 생성 및 데이터 삽입
    if "batter" in i:
        df.to_sql("batter", con=engine, index=False, if_exists="append")
        
    elif "pitcher" in i:
        df.to_sql("pitcher", con=engine, index=False, if_exists="append")

print("create sql db complete")

# sql check
with engine.connect() as conn:
    result = conn.execute("SELECT 선수명, 팀명, AVG, HR FROM batter WHERE HR >= 10 AND YEAR = 2024 ORDER BY HR DESC")
    for row in result:
        print(row)