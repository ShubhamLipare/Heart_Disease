import os
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
from src.util import save_object  # Ensure `save_object` exists

class DataTransformation:
    def __init__(self):
        pass

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

    def initiate_data_transformation_object(self, train, test):
        try:
            logging.info("Dropping 'id' column from train and test data")
            train = train.drop(columns=["id"], errors="ignore")
            test = test.drop(columns=["id"], errors="ignore")

            preprocessor = self.get_data_transformation_object()
            logging.info("Preprocessor object initiated")

            train_transformed = preprocessor.fit_transform(train)
            test_transformed = preprocessor.transform(test)
            logging.info("Transformation applied to train and test data")

            # Extract transformed column names
            transformed_columns = preprocessor.get_feature_names_out()
            train_transformed_csv = pd.DataFrame(train_transformed, columns=transformed_columns)
            test_transformed_csv = pd.DataFrame(test_transformed, columns=transformed_columns)

            # Ensure data folder exists
            os.makedirs("data", exist_ok=True)

            # Save transformed data
            train_transformed_csv.to_csv("data/train_transformed.csv", index=False)
            test_transformed_csv.to_csv("data/test_transformed.csv", index=False)
            logging.info("Transformed data is stored in 'data/' path")

            # Save the preprocessor object
            pickle_path = os.path.join("data", "preprocessor.pkl")
            save_object(pickle_path, preprocessor)
            logging.info(f"Preprocessor saved as {pickle_path}")

            return train_transformed, test_transformed
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    transformer = DataTransformation()
    train_data = pd.read_csv(r"G:\Resume projects\Heart dieses\data\train.csv")
    test_data = pd.read_csv(r"G:\Resume projects\Heart dieses\data\test.csv")
    
    train_transformed, test_transformed = transformer.initiate_data_transformation_object(train_data, test_data)

    print("Data transformation completed. Transformed files are saved in the 'data/' folder.")
