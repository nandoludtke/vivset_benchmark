import os
import json

# OpenAI
import openai
# Claude
import anthropic
# Gemini
import google.generativeai as genai

class ActionAuditor:
    def __init__(self, llm_client_type):
        allowed_llm_clients = ['openai', 'claude', 'gemini']
        if llm_client_type not in allowed_llm_clients:
            print(f"LLM client is not valid. Select one of the following: {allowed_llm_clients}")
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
    
    def audit_action_series(self, precaution: str, action_series):
        # input handling
        if not precaution or not isinstance(precaution, str):
            print("Precaution is not valid.")
            return
        if not action_series:
            print("Action series is empty.")
            return
        # compare action series to precaution
        role_msg = "You are a scientific tool designed to check if a series of actions matches a discriptive text."
        prompt_msg = f"Your task is to check if a list of actions matches the described precautions. " + \
            "Let's think step by step.\n" + \
            "Your first task is to analyze the series of actions. " + \
            "Precautions are met if the required actions happen in sequence and no other actions happen afterwards. " + \
            "Other actions not mentioned in the precautions should not be taken into account for auditing, if they take place before the precautions.\n" + \
            "Secondly, you should give recommendations if the series of actions doesn't match the precautions.\n" + \
            "The response should be given in the following format: '<REASONING>###<CORRECT/WRONG>###<RECOMMENDATIONS>'\n" + \
            f"This is the precaution text the actions should match: {precaution}\nHere is the list of actions: {action_series}"

        match self.llm_client_type:
            case 'openai':
                messages=[]
                messages.append({"role": "system", "content": role_msg})
                messages.append({"role": "user", "content": prompt_msg})
            
                # run LLM
                try:
                    result = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    ).choices[0].message.content
                    print(result)
                    return result
                except Exception as e:
                    print(f"Actions series {action_series} could not be audited.")
                    return ""
            case 'claude':
                messages = []
                messages.append({"type": "text", "text": prompt_msg})
                messages = [{"role": "user", "content": messages}]

                # run LLM
                try:
                    result = self.client.messages.create(
                        model="claude-3-opus-20240229",
                        system=role_msg,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                    ).content[0].text
                    print(result)
                    return result
                except Exception as e:
                    print(f"Actions series {action_series} could not be audited.")
                    return ""
            case 'gemini':
                messages = []
                messages.append(prompt_msg)

                # run LLM
                try:
                    result = self.client.generate_content(messages, generation_config={'temperature':0.0}).text
                    print(result)
                    return result
                except Exception as e:
                    print(f"Actions series {action_series} could not be audited.")
                    return ""
