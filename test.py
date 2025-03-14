
from sqlalchemy import create_engine,text

engine = create_engine("oracle+oracledb://sys:root@SHUBHAM:1522/oracle?mode=SYSDBA")

conn=engine.connect()


query=text("select * from sys.heart_disease ")
result=conn.execute(query).fetchall()
for rows in result:
    print(rows)


# Close connection
conn.close()
