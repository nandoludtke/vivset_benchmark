# general
import os
import sys
import json
import re
import hashlib
import pandas as pd
import random

# image processing
from base64 import b64encode, b64decode
from PIL import Image

# OpenAI
import openai
# Claude
import anthropic
# Gemini
import google.generativeai as genai


# get workspace directory
child_dir = 'src'
workspace_dir = os.getcwd()
workspace_dir = os.path.sep.join(workspace_dir.split(os.path.sep)[:workspace_dir.split(os.path.sep).index(child_dir)])

# custom imports
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.append(os.path.join(workspace_dir, 'src'))
from database.database import LocalDatabase
from yolo import yolo_labeling

class ActionInterpretation:
    def __init__(self, llm_client_type, actions_description_file: str):
        # input handling
        allowed_llm_clients = ['openai', 'claude', 'gemini']
        if llm_client_type not in allowed_llm_clients:
            print(f"LLM client is not valid. Select one of the following: {allowed_llm_clients}")
            return
        abs_actions_description_file = os.path.abspath(actions_description_file)
        if not os.path.exists(abs_actions_description_file) or not os.path.splitext(abs_actions_description_file)[1] == '.json':
            print(f"Actions description file {actions_description_file} does not exist or is not a json.")
            return
        # get init parameters
        match llm_client_type:
            case 'openai':
                self.client = openai.OpenAI()
            case 'claude':
                self.client = anthropic.Anthropic()
            case 'gemini':
                genai.configure(api_key=os.environ['GEMINI_API_KEY'])
                self.client = genai.GenerativeModel('gemini-1.5-pro')
        self.llm_client_type = llm_client_type
        with open(abs_actions_description_file, 'r') as f:
            self.actions_description = json.load(f)

    def action_series_to_json(self, action_series_str):
        # get item and action from action description
        available_actions = {}
        for item, possible_actions in self.actions_description.items():
            available_actions[item] = [action.split(':')[0] for action in possible_actions]
        
        # define messages
        role_msg = "You are a scientific tool to format a text into a list of dictionaries."
            
        prompt_msg = "Your task is to reformat a text that describes a series of actions into a list of dictionaries. " + \
            f"Use the items and actions from the following dictionary:\n{available_actions}\n" + \
            "The list should be structured like this: [{'item1': 'action1'}, {'item2': 'action2'}]. Give the list in a json format and do not give additional text. " + \
            "Here's the text to reformat:\n"

        match self.llm_client_type:
            case 'openai':
                messages=[]
                messages.append({"role": "system", "content": role_msg})
                messages.append({"role": "user", "content": prompt_msg})
                messages.append({"role": "user", "content": action_series_str})

                # run LLM
                # try:
                formatted_result = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4096,
                ).choices[0].message.content
                print(formatted_result)
                json_result = json.loads(formatted_result.split("```json")[1].split("```")[0])
                return json_result
                """except:
                    print("JSON could not be parsed.")
                    return []"""
            case 'claude':
                messages = []
                messages.append({"type": "text", "text": prompt_msg})
                messages.append({"type": "text", "text": f"{action_series_str}"})
                messages = [{"role": "user", "content": messages}]
                
                # run LLM
                # try:
                formatted_result = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    system=role_msg,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4096,
                ).content[0].text
                print(formatted_result)
                # get json format
                start = formatted_result.find('[')
                end = formatted_result.rfind(']')
                if start == -1 or end == -1 or start > end:
                    print("Could not parse json")
                    return formatted_result
                try:
                    json_result = json.loads(f"{formatted_result[start:end+1]}")
                    print(json_result)
                    return json_result
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    return formatted_result
                """except:
                    print("JSON could not be parsed.")
                    return []"""
            case 'gemini':
                messages = []
                messages.append(prompt_msg)
                messages.append(f"{action_series_str}")

                # run LLM
                # try:
                formatted_result = self.client.generate_content(messages, generation_config={'temperature':0.0}).text
                print(formatted_result)
                # get json format
                start = formatted_result.find('[')
                end = formatted_result.rfind(']')
                if start == -1 or end == -1 or start > end:
                    print("Could not parse json")
                    return formatted_result
                try:
                    json_result = json.loads(f"{formatted_result[start:end+1]}")
                    print(json_result)
                    return json_result
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    return formatted_result
                """except:
                    print("JSON could not be parsed.")
                    return []"""

    def get_action_series(self, image_series: list, item_state_baseline: dict, relevant_items: list=[]):
        # input handling
        if not image_series:
            print("No image series was passed.")
            return
        if not item_state_baseline:
            print("No image-state-baseline dict was passed.")
            return

        role_msg = "You are a scientific tool to determine changes inbetween images."
        prompt_msg = "Your task is to determine if the state of an item changes through the course of a series of images. " + \
            "Let's think step by step:\n" + \
            f"Your first task is to describe the changes of relevant items. Here is a selection of items that might be visible:\n{item_state_baseline}\n" + \
            "Use the marks and corresponding item names provided in the dictionary in your answer. Only take into account items that are within a bounding box and labeled. " + \
            "Also, you should state in between which images you observed the changes.\n" + \
            f"Secondly, based on the observed changes you should interpret them as actions that are described in the actions dictionary:\n{self.actions_description}\n" + \
            "The order of the actions should match the order of the changes regarding the images. State the images in between which the action took place.\n" + \
            "Here is the series of images to take into account:\n"

        match self.llm_client_type:
            case 'openai':
                messages=[]
                messages.append({"role": "system", "content": role_msg})
                messages.append({"role": "user", "content": prompt_msg})
                for idx, image in enumerate(image_series):
                    messages.append(
                        {"role": "user",
                            "content": [
                                {"type": "text", "text": f"Here is image {idx + 1}."},
                                {"type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64,{}".format(image['image_base64']),
                                        "detail": "high"
                                    },
                                },
                            ]
                        }
                    )

                # run LLM
                try:
                    action_series_result = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    ).choices[0].message.content
                    print(action_series_result)
                    return self.action_series_to_json(action_series_result)
                except Exception as e:
                    print(f"Action series {[image['source'] for image in image_series]} could not be interpreted")
                    return []
            case 'claude':
                messages = []
                messages.append({"type": "text", "text": prompt_msg})
                for idx, image in enumerate(image_series):
                    messages.append(
                        {
                            "type": "text",
                            "text": f"Here is image {idx + 1}."
                        }
                    )
                    messages.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image['image_base64'],
                            },
                        }
                    )
                messages = [{"role": "user", "content": messages}]
                print(messages)

                # run LLM
                try:
                    action_series_result = self.client.messages.create(
                        model="claude-3-opus-20240229",
                        system=role_msg,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    ).content[0].text
                    print(action_series_result)
                    return self.action_series_to_json(action_series_result)
                except Exception as e:
                    print(f"Action series {[image['source'] for image in image_series]} could not be interpreted")
                    return []
            case 'gemini':
                messages = []
                messages.append(prompt_msg)
                for idx, image in enumerate(image_series):
                    messages.append(f"Here is image {idx + 1}.")
                    messages.append(Image.open(image['labeled_source']))

                # run LLM
                try:
                    action_series_result = self.client.generate_content(messages, generation_config={'temperature':0.0}).text
                    print(action_series_result)
                    return self.action_series_to_json(action_series_result)
                except Exception as e:
                    print(f"Action series {[image['source'] for image in image_series]} could not be interpreted")
                    return []
