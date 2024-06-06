# general
import os
import sys
import json
import pandas as pd

# custom imports
from benchmark import pipeline

def main():
    return

# run pipeline with parameters
if __name__ == "__main__":
    arg1, arg2 = sys.argv[1:3]

    database_dir = os.path.join('environment', 'db')
    yolo_model_file = os.path.join('src', 'yolo', 'best.pt')
    actions_description_file = os.path.join('src', 'pipeline', 'actions_description.json')
    
    failed = False

    if not arg1:
        experiment = "0a"
    else:
        match arg1:
            case "baseline0":
                experiment = "0a"
            case "baseline1":
                experiment = "0b"
            case "baseline2":
                experiment = "1a"
            case _:
                print("Please pass a valid baseline name.\nHere are the available names:\nbaseline0\nbaseline1\nbaseline2")
                failed = True
            
    if not arg2:
        llm_client_type = 'openai'
    else:
        if not arg2 in ['openai', 'gemini', 'claude']:
            print("Please pass a valid LLM client type.\nHere are the available types:\nopenai\ngemini\nclaude")
            print("LLAVA is not supported in this version as it requires additional hardware.")
            failed = True
    
    if not failed:
        main()
