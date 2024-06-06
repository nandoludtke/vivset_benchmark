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

from src.database.database import LocalDatabase
from src.yolo import yolo_labeling

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
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    )
                    formatted_result = completion.choices[0].message.content
                    json_result = json.loads(formatted_result.split("```json")[1].split("```")[0])
                    return json_result
                except:
                    print("JSON could not be parsed.")
                    return []
            case 'claude':
                messages = []
                messages.append({"type": "text", "text": prompt_msg})
                messages.append({"type": "text", "text": f"{action_series_str}"})
                messages = [{"role": "user", "content": messages}]

                # run LLM
                try:
                    formatted_result = self.client.messages.create(
                        model="claude-3-opus-20240229",
                        system=role_msg,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    ).content[0].text
                    # get json format
                    start = formatted_result.find('[')
                    end = formatted_result.rfind(']')
                    if start == -1 or end == -1 or start > end:
                        print("Could not parse json")
                        return formatted_result
                    try:
                        json_result = json.loads(f"{formatted_result[start:end+1]}")
                        return json_result
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e}")
                        return formatted_result
                except:
                    print("JSON could not be parsed.")
                    return []
            case 'gemini':
                messages = []
                messages.append(prompt_msg)
                messages.append(f"{action_series_str}")

                # run LLM
                try:
                    formatted_result = self.client.generate_content(messages, generation_config={'temperature':0.0}).text
                    # get json format
                    start = formatted_result.find('[')
                    end = formatted_result.rfind(']')
                    if start == -1 or end == -1 or start > end:
                        print("Could not parse json")
                        return formatted_result
                    try:
                        json_result = json.loads(f"{formatted_result[start:end+1]}")
                        return json_result
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e}")
                        return formatted_result
                except:
                    print("JSON could not be parsed.")
                    return []

    def get_item_states(self, image, item_state_baseline: dict, relevant_items: list=[]):
        # input handling
        if not image:
            print("No image was passed.")
            return
        if not item_state_baseline:
            print("No image-state-baseline dict was passed.")
            return
        
        # set up prompts
        item_mark_list = [{item: value['mark']} for item, value in item_state_baseline.items()]
        baseline = [{item: {'label': value['mark'], 'allowed_states': value['allowed_states']}} for item, value in item_state_baseline.items()]

        role_msg = "You are a scientific tool to determine the state of items shown in an image."
        prompt_msg1 = "Your task is to determine the state of all visible items. " + \
            "Let's think step by step.\n" + \
            f"First of all you should check which labels are visible in the image from the following list:\n{item_mark_list}\n" + \
            "Here is the image:\n"
        prompt_msg2 = f"Next up you should take a look at the item that is associated with each visible label in the corresponding bounding box and determine the state of each item. " + \
            f"Here is some additional information about allowed states:\n{item_state_baseline}\n" + \
            "State all visible items and their state. For items that are not visible state 'none'."
            # f"First of all you should check which items are shown in the image. Here is a selection of items that might be visible with their unique label:\n{item_mark_list}\n" + \
            # "Only take into account items that are inside bounding boxes and labeled. " + \
            # "The hand sanitizer is considered far away, if it's close to the back of the counter. It is considered close, if it is close to the front edge of the counter. " + \
            # f"Secondly, determine the state of each item. You should use one of the states from the following reference dictionary:\n{item_state_baseline}\n" + \
            # "For all items from the reference dictionary that are not highlighted with bounding boxes and labeled, give the state 'n/a'.\n" + \
            # "Here is the image to take into account:\n"
        
        match self.llm_client_type:
            case 'openai':
                messages=[]
                messages.append({"role": "system", "content": role_msg})
                messages.append({"role": "user", "content": prompt_msg1})
                messages.append(
                    {"role": "user",
                        "content": [
                            {"type": "text", "text": "Start of image."},
                            {"type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,{}".format(image['image_base64']),
                                    "detail": "high"
                                },
                            },
                            {"type": "text", "text": "End of image."},
                        ]
                    }
                )
                messages.append({"role": "user", "content": prompt_msg2})

                # run LLM
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    )
                    state_result = completion.choices[0].message.content
                    return state_result
                except Exception as e:
                    print(e)
                    print(f"Item state of {image['source']} could not be interpreted")
                    return []
            case 'claude':
                messages = []
                messages.append({"type": "text", "text": prompt_msg})
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

                # run LLM
                try:
                    state_result = self.client.messages.create(
                        model="claude-3-opus-20240229",
                        system=role_msg,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    ).content[0].text
                    print(state_result)
                    return state_result
                except:
                    print(f"Item state of {image['source']} could not be interpreted")
                    return []
            case 'gemini':
                messages = []
                messages.append(prompt_msg)
                messages.append(Image.open(image['source']))

                # run LLM
                try:
                    state_result = self.client.generate_content(messages, generation_config={'temperature':0.0}).text
                    print(state_result)
                    return state_result
                except:
                    print(f"Item state of {image['source']} could not be interpreted")
                    return []
        

    def get_action_series(self, image_series: list, item_state_baseline: dict, relevant_items: list=[]):
        # input handling
        if not image_series:
            print("No image series was passed.")
            return
        if not item_state_baseline:
            print("No image-state-baseline dict was passed.")
            return
        
        # get states for each image
        state_series = []
        for image in image_series:
            state = self.get_item_states(
                image=image,
                item_state_baseline=item_state_baseline,
                relevant_items=relevant_items
            )
            state_series.append(state)

        # set up prompts
        role_msg = "You are a scientific tool to determine a series of actions from a series of states."
        prompt_msg = "Your task is to determine actions based on a series of states of images. " + \
            "Let's think step by step:\n" + \
            f"Your first task is to describe the changes of states of relevant items.\n" + \
            f"Secondly, based on the observed changes of states you should interpret them as actions that are described in the actions dictionary:\n{self.actions_description}\n" + \
            "The actions should be stated in the same order as the states in the context. " + \
            "Here is the series of states to take into account:\n"

        match self.llm_client_type:
            case 'openai':
                messages=[]
                messages.append({"role": "system", "content": role_msg})
                messages.append({"role": "user", "content": prompt_msg})
                for idx, state in enumerate(state_series):
                    messages.append({"role": "user", "content": f"State {idx + 1}:\n{state}"})

                # run LLM
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    )
                    action_series_result = completion.choices[0].message.content
                    print(action_series_result)
                    return self.action_series_to_json(action_series_result)
                except:
                    print(f"Action series {[image['source'] for image in image_series]} could not be interpreted")
                    return []
            case 'claude':
                messages = []
                messages.append({"type": "text", "text": prompt_msg})
                for idx, state in enumerate(state_series):
                    messages.append({"type": "text", "text": f"State {idx + 1}:\n{state}"})
                messages = [{"role": "user", "content": messages}]

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
                    return action_series_result
                except:
                    print(f"Action series {[image['source'] for image in image_series]} could not be interpreted")
                    return []
            case 'gemini':
                messages = []
                messages.append(prompt_msg)
                for idx, state in enumerate(state_series):
                    messages.append(f"State {idx + 1}:\n{state}")

                # run LLM
                try:
                    action_series_result = self.client.generate_content(messages, generation_config={'temperature':0.0}).text
                    print(action_series_result)
                    return action_series_result
                except:
                    print(f"Action series {[image['source'] for image in image_series]} could not be interpreted")
                    return []
            
        