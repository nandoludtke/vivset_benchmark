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
import google.generativeai as genai\


from src.database.database import LocalDatabase
from src.yolo import yolo_labeling

_item_state_baseline = {
    "disinfectant_wipe": {
        "mark": "A",
        "allowed_states": ["none", "on stethoscope"],
        "description": "if the disinfectant wipe (A) is laying on or close to the stethoscope, it considered on stethoscope",
        "base_state": "none"
    },
    "face_mask_box": {
        "mark": "B",
        "allowed_states": ["closed", "open"],
        "description": "If the lid of the face mask box (B) is not fully opened, it is considered closed; if the lid is of the face mask box (B) is fully opened, it is considered open",
        "base_state": "closed"
    },
    "face_shield": {
        "mark": "C",
        "allowed_states": ["far away", "close"],
        "description": "if the face shield (C) is closer in the following image than in the previous image, it considered to be close.",
        "base_state": "far away"
    },
    "gloves": {
        "mark": "D",
        "allowed_states": ["none", "on counter", "on trash can"],
        "description": "if the gloves (D) are laying on the counter, they are considered on counter; if the gloves (D) are laying on the trash can, it considered on trash can",
        "base_state": "none"
    },
    "gown": {
        "mark": "E",
        "allowed_states": ["none", "on counter"],
        "description": "if the gown (E) is laying on the counter, it is considered on counter",
        "base_state": "none"
    },
    "hand_sanitizer": {
        "mark": "F",
        "allowed_states": ["far away", "close"],
        "description": "if the hand sanitizer (F) is closer in the following image than in the previous image, it is considered to be close",
        "base_state": "far away"
    },
    "n95_box": {
        "mark": "G",
        "allowed_states": ["visible"],
        "description": "",
        "base_state": "visible"
    },
    "n95_mask": {
        "mark": "H",
        "allowed_states": ["none", "on counter"],
        "description": "if the n95 mask (H) is laying on the counter, it is considered on counter",
        "base_state": "none"
    },
    "paper_towel": {
        "mark": "J",
        "allowed_states": ["none", "on counter", "on sink handle"],
        "description": "if the paper towel (J) is laying on the counter, it is considered on counter; if the paper towel (J) is wrapped around the sink handle, it is considered on sink handle",
        "base_state": "none"
    },
    "poster": {
        "mark": "K",
        "allowed_states": ["visible"],
        "description": "",
        "base_state": "visible"
    },
    "sink_handle": {
        "mark": "L",
        "allowed_states": ["visible"],
        "description": "",
        "base_state": "visible"
    },
    "stethoscope": {
        "mark": "M",
        "allowed_states": ["visible"],
        "description": "",
        "base_state": "visible"
    },
    "trash_can": {
        "mark": "N",
        "allowed_states": ["visible"],
        "description": "",
        "base_state": "visible"
    },
    "water_stream": {
        "mark": "O",
        "allowed_states": ["none", "visible"],
        "description": "If the water strem (O) is not visible, the sink handle is considered off. If the water stream (O) is visible, the sink handle is considered on.",
        "base_state": "none"
    }
}

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
                
    def get_action(self, image_pair: list, item_state_baseline: dict, relevant_items: list=[]):
        # input handling
        if not len(image_pair) == 2:
            print("No image pair was passed.")
            return
        if not item_state_baseline:
            print("No image-state-baseline dict was passed.")
            return
        
        
        item_labels = [{item: state['mark']} for item, state in item_state_baseline.items()]
        # item_state_baseline = [{item: {'label': state['mark'], 'allowed_states': state['allowed_states'], 'description': state['description']}} for item, state in item_state_baseline.items()]
        
        role_msg = "You are a scientific tool to determine changes of items inbetween images."
        
        prompt_msg1 = "Your task is to determine which labels are visible in the reference image. " + \
            "Let's think step by step:\n" + \
            f"Your first task is to inspect the reference image and check which labels are visible that can be found in the label list:\n{item_labels}\n" + \
            "State all visible labels in the reference image in a list. " + \
            "Here is the reference image:\n"
        prompt_msg2 = f"Your second task is to inpsect the comparison image and check which labels are visible that can be found in the label list:\n{item_labels}\n" + \
            "State all visible labels in the comparison image in a list. " + \
            "Here is the comparison image:\n"
        
        prompt_msg3 = "As a third step take a look at the items inside the bounding boxes that are associated with the labels that you found in each image. " + \
            "State all visible items and their state in the reference and comparison image in a list seperately. " + \
            f"Describe their state and how their state changes according to the following reference dictionary:\n{_item_state_baseline}\n" + \
            "State the visible change that is the most prominent using the following format: 'CHANGE: <change description>'"
            # f"Interpret the most prominent change as an action based on the following descriptions:\n{self.actions_description}\n" + \
            # "State the action using the following format 'ACTION: <action description>'. "

        # prompt_msg1 = "Your task is to interpret changes in between two images as actions.\n" + \
        #     "Let's think step by step." + \
        #     f"Your first task is to check which labels from the following list are visible in the reference image:\n{item_labels}\n" + \
        #     "Here is the reference image:\n"
        # prompt_msg2 = f"Secondly, you should check which labels from the following list are visible in the comparison image:\n{item_labels}\n" + \
        #     "Here is the comparison image:\n"
        # prompt_msg3 = f"As a third step you should take a look at the visible labels and their associated items in corresponding bounding boxes and compare their states in the reference and comparison image. " + \
        #     f"For evaluating the state compare the visibility, position related to the camera and status in the reference and comparison images. Here is a reference dictionary rearding item states:\n{_item_state_baseline}\n"
        # prompt_msg3 = f"As a third step, determine how the items that are in bounding boxes associated with their labels change in state. Here is a reference dictionary that gives you more information:\n{_item_state_baseline}\n"
        
            # f"As a third step you should take a look at the visible labels and their associated items in corresponding bounding boxes and determine their states. " + \
            # f"For evaluating the state compare the visibility, check if they move or their appearance changes in the reference and comparison images. Here is a reference dictionary that gives you more information:\n{_item_state_baseline}\n" + \
            # "Here are some examples for a change in state:\n" + \
            # "Face mask box (C) was closed in reference image but appears open as the lid is opened in comparison image.\n" + \
            # "Hand sanitizer (F) was far away in reference image but appears closer in comparison image.\n" + \
            # "N95 mask (H) was not visible in reference image but is on counter in comparison image.\n"
            
            # f"As a last step, the changes in states should be interpreted as action based on the following descriptions:\n{self.actions_description}\n" + \
            # "State the interpreted action that you are most certain about using the following format: 'ACTION: <action description>'" + \
            # "If no action takes place state 'ACTION: none'"

        prompt_msg1 = "Your task is to determine which labels are visible in the reference image. " + \
            "Let's think step by step:\n" + \
            f"Your first task is to inspect the reference image and check which labels are visible that can be found in the label list:\n{item_labels}\n" + \
            "State all visible labels in the reference image in a list. " + \
            "Here is the reference image:\n"
        prompt_msg2 = f"Your second task is to inspect the comparison image and check which labels are visible that can be found in the label list:\n{item_labels}\n" + \
            "State all visible labels in the comparison image in a list. " + \
            "Here is the comparison image:\n"
        
        prompt_msg3 = "As a third step take a look at both images again and see if items which can be found in the bounding box of each label that are visible in both images change in their state. Items may change in visibility, move to a different position on the counter or might change in state such as 'closed' or 'open'. " + \
            "Describe how the items change from the reference image to the comparison image. " + \
            f"Based on your findings interpret the most prominent changes as actions. Here is a list of available actions:\n{self.actions_description}\n" + \
            "State the most prominent action at the end of your answer in the following format: 'ACTION: put on n95 mask'. Only state one action."
            
            # the items inside the bounding boxes that are associated with the labels that you found in each image. " + \
            # "State all visible items and their state in the reference and comparison image in a list seperately. " + \
            # "If an item is visible in both images compare the position and appearance in both images and describe if these characteristics change. " + \
            # f"Take into account the following dictionary with more information about possible states:\n{_item_state_baseline}\n" + \
            # "State all visible changes."
            
            # "Also, if a labeled item from the reference image is also visible in the comparison image describe how the appearance or position changes from the reference image to the comparison image. " + \
            # "To determine the state of each item take a look at both images to see if the states differ and if this gives you information about each state such as appearance, position or visibility. " + \
            # f"Describe their state and how their state changes according to the following reference dictionary:\n{_item_state_baseline}\n" + \
            
            
            

            

            # f"Based on the visible labels take a look on the associated items found in the corresponding bounding boxes. Interpret the items' state. Check how the states change inbetween the two images." + \
            
            # the state of each visible associated item shown in a bounding box. How do the states changein between the images. Here is a dictionary with information about possible states:\n{item_state_baseline}\n" + \
          
        
        match self.llm_client_type:
            case 'openai':
                messages=[]
                messages.append({"role": "system", "content": role_msg})
                messages.append({"role": "user", "content": prompt_msg1})
                messages.append(
                    {"role": "user",
                        "content": [
                            {"type": "text", "text": "Start of reference image:"},
                            {"type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,{}".format(image_pair[0]['image_base64']),
                                    "detail": "high"
                                },
                            },
                            {"type": "text", "text": "End of reference image:"},
                        ]
                    }
                )
                messages.append({"role": "user", "content": prompt_msg2})
                messages.append(
                    {"role": "user",
                        "content": [
                            {"type": "text", "text": "Start of comparison image:"},
                            {"type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,{}".format(image_pair[1]['image_base64']),
                                    "detail": "high"
                                },
                            },
                            {"type": "text", "text": "End of comparison image:"},
                        ]
                    }
                )
                messages.append({"role": "user", "content": prompt_msg3})
                print(f"Image 1: {image_pair[0]['labeled_source']}")
                print(f"Image 2: {image_pair[1]['labeled_source']}")

                # run LLM
                try:
                    action_series_result = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    ).choices[0].message.content
                    print(action_series_result)
                    return action_series_result
                except Exception as e:
                    print(e)
                    print(f"Action series {[image['source'] for image in image_pair]} could not be interpreted")
                    return 'No action could be found in between images.'
            case 'claude':
                messages = []
                messages.append({"type": "text", "text": prompt_msg})
                for idx, image in enumerate(image_pair):
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
                    return action_series_result
                except Exception as e:
                    print(e)
                    print(f"Action series {[image['source'] for image in image_pair]} could not be interpreted")
                    return 'No action could be found in between images.'
            case 'gemini':
                messages = []
                messages.append(prompt_msg)
                for idx, image in enumerate(image_pair):
                    messages.append(f"Here is image {idx + 1}.")
                    messages.append(Image.open(image['labeled_source']))
                # run LLM
                try:
                    action_series_result = self.client.generate_content(messages, generation_config={'temperature':0.0}).text
                    print(action_series_result)
                    return action_series_result
                except Exception as e:
                    print(e)
                    print(f"Action series {[image['source'] for image in image_pair]} could not be interpreted")
                    return 'No action could be found in between images.'

    def get_action_series(self, image_series: list, item_state_baseline: dict, relevant_items: list=[]):
        # input handling
        if not image_series:
            print("No image series was passed.")
            return
        if not item_state_baseline:
            print("No image-state-baseline dict was passed.")
            return
        
        action_series = []
        for idx in range(1, len(image_series)):
            action_result = self.get_action(
                image_pair=image_series[idx - 1:idx + 1],
                item_state_baseline=item_state_baseline,
                relevant_items=relevant_items)
            action_idx = action_result.rfind('ACTION:')
            if action_idx == -1:
                action_result = "none"
            else:
                action_result = action_result[action_idx:]
            action_series.append(action_result)
            print([image['labeled_source'] for image in image_series[idx - 1:idx + 1]])
        
        # run LLM
        try:
            print(action_series)
            return self.action_series_to_json(action_series)
        except Exception as e:
            print(f"Action series {[image['source'] for image in image_series]} could not be interpreted")
            return []
