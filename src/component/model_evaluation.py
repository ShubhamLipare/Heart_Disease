import os
import sys
import mlflow
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.util import load_object

class ModelEvaluator:
    def __init__(self, model_path: str, test_data):
        self.model_path = model_path
        self.x_test, self.y_test = test_data[:, :-1], test_data[:, -1]
        
    def evaluate_model(self):
        try:
            logging.info("Loading the best model for evaluation")
            model = load_object(self.model_path)
            
            y_pred = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            logging.info("Logging evaluation metrics to MLflow")
            with mlflow.start_run():
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_dict(report, "classification_report.json")
                mlflow.log_dict({"confusion_matrix": conf_matrix.tolist()}, "confusion_matrix.json")
                mlflow.sklearn.log_model(model, "best_model")
            
            pd.DataFrame(report).transpose().to_csv("data/evaluation_report.csv", index=True)
            logging.info("Evaluation report saved successfully")
            
            return accuracy, report, conf_matrix
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    test_data = "path_to_test_data.npy"  # Replace with actual test data path
    evaluator = ModelEvaluator("data/model.pkl", test_data)
    accuracy, report, conf_matrix = evaluator.evaluate_model()
    print("Evaluation Complete. Accuracy:", accuracy)
