import os
import json
from copy import deepcopy
from .utils import get_chatbot_response, extract_json_block, check_json_output
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class GuardAgent():
    def __init__(self):
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=os.getenv('RUNPOD_TOKEN'),
            base_url=os.getenv('RUNPOD_CHATBOT_URL')
        )
        self.model_name = os.getenv('MODEL_NAME')
    
    def get_response(self, messages):
        """
        Determines if the user's message is relevant to the coffee shop.
        Returns a structured response indicating if the input is 'allowed' or 'not allowed'.
        """
        messages = deepcopy(messages)

        system_prompt = """
            You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
            Your task is to determined whether to user is asking something relevant to the coffee shop or not.
            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menu items and coffee shop related questions.
            2. Ask questions about menu items, they can ask for ingredients in an item and more details about the item.
            3. Make an order.
            4. Ask about recommendation of what to buy

            The user is not allowed to:
            1. Ask questions about anything else other than our coffee shop.
            2. Ask questions about the staff or how to make a certain menu items.

            Your output should be in a structured json format like so. Each key is a string and each value is a string.
            Make sure to follow the format exactly:
            {
            "chain of thought": "go over each of the points above and see if the message lies under this point or not. Then you write some thoughts about what point is this input relevant to.",
            "decision": "'allowed' or 'not allowed'. Pick one of those and only write the word.",
            "message": "leave the message empty '' if it is allowed, otherwise write 'Sorry, I can't help with that. Can I help you with your order?'"
            }
        """

        input_message = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output = get_chatbot_response(self.client, self.model_name, input_message)
        chatbot_output = check_json_output(self.client, self.model_name, chatbot_output)
        output = self.postprocess(chatbot_output)

        return output
    
    def postprocess(self, output):
        """
        Converts raw model output (JSON string) into the expected response format
        for the chatbot system.
        """
        clean_output = extract_json_block(output)
        output = json.loads(clean_output)

        dict_output = {
            "role": "assistant",
            "content": output['message'],
            "memory": {
                "agent":"guard_agent",
                "guard_decision":output['decision']
            }
        }
        return dict_output