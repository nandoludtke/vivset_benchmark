# general imports
import os
import sys
import json
import pandas as pd
import argparse

from src.benchmark import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', nargs='?', type=str)
parser.add_argument('--llm', nargs='?', type=str)
parser.add_argument('--rooms', nargs='*', type=str)
parser.add_argument('--precautions', nargs='*', type=str)
parser.add_argument('--iterations', nargs='?', type=int)

def main(database_dir, yolo_model_file, actions_description_file, experiment, llm_client_type, rooms, precautions, N_iterations):

    audit_pipeline = pipeline.ImageSeriesAuditingPipeline(
        database_dir=database_dir,
        actions_description_file=actions_description_file,
        llm_client_type=llm_client_type,
        experiment=experiment,
        yolo_model_file=yolo_model_file
    )

    # iterate through rooms and precautions
    baseline = 'baseline0'
    match experiment:
        case '0a':
            baseline = 'baseline0'
        case '0b':
            baseline = 'baseline1'
        case '1a':
            baseline = 'baseline2'
    for room in rooms:
        room_dir = os.path.join('environment', room)
        if not os.path.exists(room_dir) or not os.path.isdir(room_dir):
            print(f"Room directory {room_dir} was not found.")
            continue
        for precaution in precautions:
            precaution_dir = os.path.join(room_dir, precaution)
            if not os.path.exists(precaution_dir) or not os.path.isdir(precaution_dir):
                print(f"Precaution directory {precaution_dir} was not found.")
                continue
            # iterate through test series files
            test_series_dir = os.path.join('environment', 'test_sequences', precaution, 'generated')
            for test_series in os.listdir(test_series_dir):
                if not precaution in test_series or not os.path.splitext(test_series)[1] == '.txt':
                    print(f"Test series file {test_series} is wrong data type or name is not valid.")
                    continue
                # get image name list
                test_series_file = os.path.join(test_series_dir, test_series)
                with open(test_series_file, 'r') as f:
                    image_name_list = f.read().split('\n')
                # run pipeline multiple N times and evaluate
                columns = ['Test Series', 'Iteration', 'Label Result', 'Action Result', 'Audit Result', 'Overall Evaluation']
                evaluation_df = pd.DataFrame(columns=columns)
                for iteration in range(0, N_iterations):
                    label_result, action_result, audit_result = audit_pipeline.run_pipeline(
                        room=room,
                        precaution=precaution,
                        image_name_list=image_name_list
                    )
                    # evaluate results
                    test_series_name = os.path.splitext(test_series)[0]
                    audit_result_evaluation = False
                    if audit_result:
                        audit_reasoning = ""
                        audit_evaluation = ""
                        # audit_recommendation = ""
                        try:
                            audit_reasoning = audit_result.split('###')[0]
                            audit_evaluation = audit_result.split('###')[1]
                            # audit_recommendation = audit_result.split('###')[2]
                            if ('correct' in audit_evaluation.lower() and '-p' in test_series_name) or ('wrong' in audit_evaluation.lower() and '-n' in test_series_name):
                                audit_result_evaluation = True
                        except:
                            print(f"Could not parse answer: {audit_result}")
                        
                    row = [test_series_name, iteration + 1, label_result, action_result, audit_result, audit_result_evaluation]
                    evaluation_df.loc[len(evaluation_df)] = row
                # save evaluation dataframe in test series directory
                save_dir = os.path.join('environment', 'test_sequences', precaution, 'results')
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(os.path.join(save_dir, baseline), exist_ok=True)
                evaluation_df.to_csv(os.path.join(save_dir, baseline, f"{room}_{test_series_name}_{llm_client_type}_evaluation.csv"))
    return

if __name__ == "__main__":
    args = parser.parse_args()
    baseline = args.baseline
    llm_client_type = args.llm
    rooms = args.rooms
    precautions = args.precautions
    N_iterations = args.iterations

    # check input values
    failed = False
    if not baseline in ['baseline0', 'baseline1', 'baseline2']:
        print("You can pass a valid baseline.\nHere are the available names:\nbaseline0\nbaseline1\nbaseline2")
        print("Using default baseline0")
        baseline = 'baseline0'
    if not llm_client_type in ['openai', 'gemini', 'claude']:
        print("You can pass a valid LLM type.\nHere are the available LLM types:\nopenai\ngemini\nclaude")
        print("LLAVA is not supported due to hardware restricitons.")
        print("Using default openai")
        llm_client_type = 'openai'
    for room in rooms:
        if not room in ['surgery', 'icu', 'labour']:
            print(f"Room '{room}' is not valid and will be removed.\nHere are the available rooms:\nsurgery\nicu\nlabour")
            room = []
    rooms = list(filter(None, rooms))
    if not rooms:
        print("Using default surgery")
        rooms = ['surgery']
    for precaution in precautions:
        if not precaution in ['contact', 'contact_plus', 'droplet', 'neutropenia', 'covid']:
            print(f"Precaution '{precaution}' is not valid and will be removed.\nHere are the available precautions:\ncontact\ncontact_plus\ndroplet\nneutropenia\ncovid")
            precaution = []
    precautions = list(filter(None, precautions))
    if not precautions:
        print("Using default contact")
        precautions = ['contact']
    if not N_iterations:
        print("Passed number of iterations is not valid.\nThe number of iterations has to be an integer greater than 0")
        print("Using default 1")
        N_iterations = 1

    # refactor baseline to experiment
    experiment = '0a'
    match baseline:
        case 'baseline0':
            experiment = '0a'
        case 'baseline1':
            experiment = '0b'
        case 'baseline2':
            experiment = '1a'
    
    # run pipeline
    if not failed:
        database_dir = os.path.join('environment', 'db')
        yolo_model_file = os.path.join('src', 'yolo', 'best.pt')
        actions_description_file = os.path.join('src', 'benchmark', 'actions_description.json')

        print("Running pipeline with following parameters:")
        print(f"Baseline: {baseline}")
        print(f"LLM type: {llm_client_type}")
        print(f"Rooms: {rooms}")
        print(f"Precautions: {precautions}")
        print(f"Number of iterations: {N_iterations}")
        print(f"Database directory: {database_dir}")
        print(f"YOLO model file: {yolo_model_file}")
        print(f"Actions description file: {actions_description_file}")

        main(database_dir, yolo_model_file, actions_description_file, experiment, llm_client_type, rooms, precautions, N_iterations)