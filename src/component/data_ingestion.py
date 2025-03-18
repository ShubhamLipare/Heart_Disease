from sqlalchemy import create_engine,text
import pandas as pd
from dotenv import load_dotenv
import os
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.model_selection import train_test_split

class DataInjestionConfig:
    def __init__(self,source_data_path:str=os.path.join("artifacts","source.csv"),
                 train_data_path:str=os.path.join("artifacts","train.csv"),
                 test_data_path:str=os.path.join("artifacts","test.csv")):
        self.source_data_path=source_data_path
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path


class DataInjestion:
    def __init__(self):
        self.data_injestion_config=DataInjestionConfig()
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

            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)

            os.makedirs(os.path.dirname(self.data_injestion_config.source_data_path),exist_ok=True)
            logging.info("artifacts folder created")

            df.to_csv(self.data_injestion_config.source_data_path,index=False,header=True)
            logging.info(f"source data is stored at {self.data_injestion_config.source_data_path}")

            train_data.to_csv(self.data_injestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_injestion_config.test_data_path,index=False,header=True)

            return (self.data_injestion_config.train_data_path,self.data_injestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
