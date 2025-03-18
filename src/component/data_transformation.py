import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
from src.util import save_object  # Ensure `save_object` exists

class DataTransformationConfig:
    def __init__(self,preprocessor_path:str=os.path.join("artifacts","preprocessor.pkl"),
                 x_train_transformed_path:str=os.path.join("artifacts","x_train_transformed.csv"),
                 x_test_transformed_path:str=os.path.join("artifacts","x_test_transformed.csv")):
        self.preprocessor_path=preprocessor_path
        self.x_train_transformed_path=x_train_transformed_path
        self.x_test_transformed_path=x_test_transformed_path

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        encoding_col = ['cp', 'restecg', 'thal', "slope", "ca"]
        continuous_var = ["age", "trestbps", "chol", 
                          "thalach", "oldpeak"]

        encoding_pipeline = Pipeline(steps=[
           ("one_hot_encoding", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])

        scaling_pipeline = Pipeline(steps=[
            ("standard_scaler", StandardScaler())
        ])

        preprocessing = ColumnTransformer(transformers=[
            ("encode", encoding_pipeline, encoding_col),
            ("scaler", scaling_pipeline, continuous_var)
        ], remainder="passthrough")  # Keep other numerical columns

        return preprocessing

    def initiate_data_transformation_object(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test csv files")

            logging.info("Dropping 'id' column from train and test data")
            train_df = train_df.drop(columns=["id"], errors="ignore")
            test_df = test_df.drop(columns=["id"], errors="ignore")

            preprocessor = self.get_data_transformation_object()
            logging.info("Preprocessor object initiated")

            x_train,y_train=train_df.drop("target",axis=1),train_df["target"]
            x_test,y_test=test_df.drop("target",axis=1),test_df["target"]

            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed = preprocessor.transform(x_test)
            logging.info("Transformation applied to train and test data")

            # Extract transformed column names
            transformed_columns = preprocessor.get_feature_names_out()
            train_transformed_csv = pd.DataFrame(x_train_transformed, columns=transformed_columns)
            test_transformed_csv = pd.DataFrame(x_test_transformed, columns=transformed_columns)

            os.makedirs(os.path.dirname(self.data_transformation_config.x_train_transformed_path), exist_ok=True)#creating artifacts folder
            # Save transformed data
            train_transformed_csv.to_csv(self.data_transformation_config.x_train_transformed_path, index=False)
            test_transformed_csv.to_csv(self.data_transformation_config.x_test_transformed_path, index=False)
            logging.info("Transformed data is stored in 'artifacts' path")

            # Save the preprocessor object
            save_object(self.data_transformation_config.preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved at {self.data_transformation_config.preprocessor_path}")

            return (x_train_transformed,y_train,
                    x_test_transformed,y_test)
        
        except Exception as e:
            raise CustomException(e,sys)

