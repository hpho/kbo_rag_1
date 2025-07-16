from sqlalchemy import create_engine
engine = create_engine("sqlite:///kbo_stats.db")

with engine.connect() as conn:
    result = conn.execute("SELECT 선수명, 팀명, AVG, HR FROM batter WHERE HR >= 10 AND YEAR = 2024 ORDER BY HR DESC")
    for row in result:
        row
        
if row[0] == "서호철":
    print(True)