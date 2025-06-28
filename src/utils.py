#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

from datetime import datetime
import logging
import os
import pathlib
import shutil
import sys
import time
import yaml


def clear_logs(config: dict):


    path_logs = config['log_dir']
    if os.path.exists(path_logs):
        shutil.rmtree(path_logs)
        os.makedirs(path_logs)
    else:
        logging.warning("No logs directory found!")



def create_logger(name:str, logs_folder: str) -> logging.Logger:


    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    timestamp = datetime.today().strftime("%Y-%M-%d %HH.%MM.%SS")
    log_file = f"{name}_{timestamp}.log"
    log_path = os.path.join(logs_folder, log_file)

    # create logger
    logging.basicConfig(filename=log_path, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)

    return logger






