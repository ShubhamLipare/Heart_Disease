from sqlalchemy import create_engine,text
import pandas as pd
from dotenv import load_dotenv
import os
from src.exception import CustomException
from src.logger import logging
import sys

class DataInjestion:
    def __init__(self):
        load_dotenv()
        self.user=os.getenv("USER")
        self.password=os.getenv("PASSWORD")
        self.host=os.getenv("HOST")
        self.port=os.getenv("PORT")
        self.schema=os.getenv("SCHEMA")
    
    def connect_to_db(self):
        try:
            self.engine = create_engine(f"oracle+oracledb://{self.user}:{self.password}@{self.host}:{self.port}/oracle?mode={self.schema}")
            logging.info("connected to db")
        except Exception as e:
            raise CustomException(e,sys)
        
    def fetch_data(self):
        try:
            with self.engine.connect() as conn:
                query=text("select * from heart_disease")
                df=pd.read_sql(query,conn)
            logging.info("fetched data from table")

            os.makedirs("data",exist_ok=True)
            with open("data/source.csv","w") as file:
                df.to_csv(file,index=False)
            logging.info("source data is stored in /data path")
            return df
        except Exception as e:
            raise CustomException(e,sys)



if __name__=="__main__":
    injestion=DataInjestion()
    injestion.connect_to_db()
    df=injestion.fetch_data()
    print(df.head())