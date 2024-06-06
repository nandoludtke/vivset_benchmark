# general
import os
import sys
import json
import re
import hashlib
import pandas as pd
import random

# image processing
from base64 import b64encode


# custom imports
from src.database import database
# STAGE 1 imports
from src.yolo import yolo_labeling
# STAGE 2 imports
from src.benchmark import baseline0_image_series_changes
from src.benchmark import baseline1_image_pair_changes
from src.benchmark import baseline2_image_series_states
# STAGE 3 imports
from src.benchmark import action_auditing

class ImageSeriesAuditingPipeline:
    def __init__(self, database_dir: str, actions_description_file: str, llm_client_type: str='openai', experiment: str="0a", yolo_model_file: str=os.path.join('src', 'yolo', 'best.pt'), skip_labeling=False):
        # input handling
        if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
            print(f"No valid database path was passed: {database_dir}")
            return
        if not os.path.exists(yolo_model_file) or not os.path.splitext(yolo_model_file)[1].lower() == '.pt':
            print(f"No YOLO model file found at {yolo_model_file}")
            return
        if not os.path.exists(actions_description_file) or not os.path.splitext(actions_description_file)[1].lower() == '.json':
            print(f"No actions description file found at {actions_description_file}")
            return
        allowed_llm_clients = ['openai', 'claude', 'gemini']
        if llm_client_type not in allowed_llm_clients:
            print(f"LLM client is not valid. Select one of the following: {allowed_llm_clients}")
            return
        if not experiment:
            print("No experiment was passed.")
            return
        # define relevant items
        self.relevant_items = [
            'disinfectant_wipe',
            'face_mask_box',
            'gloves',
            'gown',
            'hand_sanitizer',
            'paper_towel',
            'poster',
            'stethoscope',
            'trash_can',
            'water_stream',
            'n95_mask',
            'face_shield'
        ]
        # get database
        self.database = database.LocalDatabase(
            db_dir=database_dir
        )
        self.database.get_existing_collection()
        # get label predictor
        self.label_predictor = yolo_labeling.LabelPredictor(
            model_path=yolo_model_file,
            relevant_items=self.relevant_items,
            skip=skip_labeling
        )
        # get LLM client
        self.llm_client_type = llm_client_type
        # get action interpreter for experiment
        self.experiment = experiment
        match self.experiment:
            case '0a':
                self.action_interpreter = baseline0_image_series_changes.ActionInterpretation(
                    llm_client_type=self.llm_client_type,
                    actions_description_file=actions_description_file
                )
            case '0b':
                self.action_interpreter = baseline1_image_pair_changes.ActionInterpretation(
                    llm_client_type=self.llm_client_type,
                    actions_description_file=actions_description_file
                )
            case '1a':
                self.action_interpreter = baseline2_image_series_states.ActionInterpretation(
                    llm_client_type=self.llm_client_type,
                    actions_description_file=actions_description_file
                )
        # get action auditor
        self.action_auditor = action_auditing.ActionAuditor(
            llm_client_type=self.llm_client_type
        )

    def run_pipeline(self, room: str, precaution: str, image_name_list: list):
        # input handling
        allowed_rooms = ['surgery', 'icu', 'labour']
        allowed_precautions = ['contact', 'contact_plus', 'covid', 'droplet', 'neutropenia']
        if not room in allowed_rooms:
            print(f"Room {room} was not allowed. Allowed rooms: {allowed_rooms}")
            return
        if not precaution in allowed_precautions:
            print(f"Precaution {precaution} was not allowed. Allowed precautions: {allowed_precautions}")
            return

        # get precaution text
        precaution_file = os.path.join('environment', 'precautions', f"{precaution}.txt")
        if not os.path.exists(precaution_file):
            print(f"Precaution file could not be found in {precaution_file}")
        with open(precaution_file, 'r') as f:
            precaution_text = f.read()

        # get image series from image name list
        image_series = []
        for image_name in image_name_list:
            # retrieve image from database
            search_id = f"{room}_{precaution}_{image_name}"
            retrieved_image = self.database.get_image_by_id(image_id=search_id)
            if not retrieved_image['ids']:
                print(f"Image {image_name} oculd not be retrieved from database.")
                return
            # append image id, path and metadata to image series
            image_series.append(
                {
                    "id": search_id,
                    "source": retrieved_image['documents'][0],
                    "state": retrieved_image['metadatas'][0]
                }
            )

        # STAGE 1: predict labels
        label_result = self.label_predictor.label_images(
            images_list=[image["source"] for image in image_series]
        )
        # encode labeled images
        for image in image_series:
            labeled_image_file = os.path.join(os.path.dirname(image['source']), 'temp', os.path.basename(image['source']))
            if os.path.exists(labeled_image_file) and os.path.splitext(labeled_image_file)[1].lower() in ['.jpeg', '.jpg', '.png']:
                with open(labeled_image_file, "rb") as f:
                    image["image_base64"] = b64encode(f.read()).decode('utf-8')
                image["labeled_source"] = labeled_image_file
        
        # STAGE 2: interpret actions in image series
        action_series_result = self.action_interpreter.get_action_series(
            image_series=image_series,
            item_state_baseline=self.database.item_state_baseline,
            relevant_items=self.relevant_items
        )
        # STAGE 3: audit action series
        action_audit_result = self.action_auditor.audit_action_series(
            precaution=precaution_text,
            action_series=action_series_result
        )

        return label_result, action_series_result, action_audit_result