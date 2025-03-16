import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.util import save_object,evaluate_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.component.data_transformation import DataTransformation

class ModelTrainerConfig:
    def __init__(self,trained_model_file_path:str=os.path.join("data","model.pkl"),
                 model_train_report_path:str=os.path.join("data","model_trainer_report.csv"),
                 model_test_report_path:str=os.path.join("data","model_test_report.csv")):
        
        self.trained_model_file_path=trained_model_file_path
        self.model_train_report_path=model_train_report_path
        self.model_test_report_path=model_test_report_path

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            x_train, x_test, y_train, y_test = train_arr[:, :-1], test_arr[:, :-1], train_arr[:, -1], test_arr[:, -1]
            logging.info("Train-test split completed")

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Classifier (SVC)": SVC(),
                "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30, 40],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [10, 50, 100],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Support Vector Classifier (SVC)": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                "K-Nearest Neighbors (KNN)": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }

            train_report, test_report, best_models = evaluate_model(x_train, x_test, y_train, y_test, models, params)

            # Save reports
            pd.DataFrame.from_dict(train_report, orient="index", columns=["accuracy"]).to_csv(self.model_trainer_config.model_train_report_path)
            pd.DataFrame.from_dict(test_report, orient="index", columns=["accuracy"]).to_csv(self.model_trainer_config.model_test_report_path)
            logging.info("Stored model reports in data folder")

            best_model_name = max(test_report, key=test_report.get)
            best_model_score = test_report[best_model_name]
            best_model = best_models[best_model_name]  # Ensure we are using the trained model

            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy.")

            # Save the best model
            save_object(filepath=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Ensure model is trained before prediction
            best_model.fit(x_train, y_train)

            # Make predictions
            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    model_trainer=ModelTrainer()
    transformer=DataTransformation()
    train_data = pd.read_csv(r"G:\Resume projects\Heart dieses\data\train.csv")
    test_data = pd.read_csv(r"G:\Resume projects\Heart dieses\data\test.csv") 
    train_arr, test_arr = transformer.initiate_data_transformation_object(train_data, test_data)
    score=model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(score)