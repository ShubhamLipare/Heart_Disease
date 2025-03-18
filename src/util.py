import pickle
import os
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info(f"Picke file is stored at {filepath}")
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train, x_test, y_train, y_test, models, params):
    try:
        train_report = {}
        test_report = {}
        best_models = {}

        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Heart Disease Prediction")

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            if model_name in params:
                param_grid = params[model_name]
            else:
                logging.warning(f"No hyperparameters specified for {model_name}. Using default.")
                param_grid = {}

            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
            grid_search.fit(x_train, y_train)
            logging.info(f"{model_name} fitted using GridSearchCV")

            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)
            logging.info(f"Predictions completed for {model_name}")

            train_accuracy=accuracy_score(y_train, y_train_pred)
            test_accuracy=accuracy_score(y_test, y_test_pred)

            train_report[model_name] = train_accuracy
            test_report[model_name] = test_accuracy

            with mlflow.start_run():
                mlflow.sklearn.log_model(best_model,model_name)
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)


            logging.info(f"{model_name} - Train Accuracy: {train_report[model_name]}, Test Accuracy: {test_report[model_name]}")

        return train_report, test_report, best_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(path):
    try:
        with open(path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)



