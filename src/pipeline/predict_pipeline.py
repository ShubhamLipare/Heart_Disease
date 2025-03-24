import os
import sys
import pandas as pd
from src.util import load_object
from src.logger import logging
from src.exception import CustomException
class PredictPipeline:
    def __init__(self):
        self.preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
        self.model_path=os.path.join("artifacts","model.pkl")

    def predict(self,df):
        try:
            preprocessor=load_object(self.preprocessor_path)
            model=load_object(self.model_path)
            logging.info("Read preprocessor and model pickel file")
            input_data_arr=preprocessor.transform(df)
            prediction=model.predict(input_data_arr)
            logging.info(f"Prediction:{prediction}")
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
