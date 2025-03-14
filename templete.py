import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s')

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/component/__init__.py",
    f"src/component/data_ingestion.py",
    f"src/component/data_transformation.py",
    f"src/component/model_trainer.py",
    f"src/pipeline/__init__.py",
    f"src/pipeline/predict_pipeline.py",
    f"src/pipeline/train_pipeline.py",
    f"src/util.py",
    f"src/logger.py",
    f"src/exception.py",

    "config/config.yaml",
    "params.yaml",
    "main.py",
    "Dockerfile",
    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating filedir:{filedir} for file:{filename}")

    if not filepath.exists() or os.path.getsize(filepath)==0:
        filepath.touch()
        logging.info(f"Created filepath:{filepath}")
    
    else:
        logging.info(f"{filename} already exist")
