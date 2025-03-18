from src.component.data_ingestion import DataInjestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import sys
import os



def main():
    try:
        data_injestion=DataInjestion()
        data_injestion.connect_to_db()
        train_path,test_path=data_injestion.fetch_data()
        logging.info("returned train and test paths")

        data_trasformation=DataTransformation()
        x_train_transformed,y_train,x_test_transformed,y_test=data_trasformation.initiate_data_transformation_object(train_path,test_path)
        logging.info("retuned x_train,y_train,x_test,y_test")

        model_trainer=ModelTrainer()
        model_trainer.initiate_model_trainer(x_train_transformed,y_train,x_test_transformed,y_test)

    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    main()