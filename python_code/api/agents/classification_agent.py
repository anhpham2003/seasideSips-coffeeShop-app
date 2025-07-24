import os
import json
from copy import deepcopy
from .utils import get_chatbot_response, extract_json_block, check_json_output
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class ClassificationAgent():
    def __init__(self):
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key = os.getenv('RUNPOD_TOKEN'),
            base_url = os.getenv('RUNPOD_CHATBOT_URL')
        )
        self.model_name = os.getenv('MODEL_NAME')

    def get_response(self, messages):
        """
        Determines which specialized agent should handle the user's input.
        The assistant classifies the message into one of:
        - details_agent
        - order_taking_agent
        - recommendation_agent
        """
        messages = deepcopy(messages)

        system_prompt = """
            You are a helpful AI assistant for a coffee shop application.
            Your task is to determine what agent should handle the user input. You have 3 agents to choose from:

            1. details_agent: Responsible for answering questions about the coffee shop, like location, delivery, hours, menu items, or what items we have.
            2. order_taking_agent: Responsible for taking orders and having a conversation with the user until the order is completed.
            3. recommendation_agent: Responsible for suggesting what the user should buy based on their preferences or questions.

            Your output must be in a strict JSON format. Each key and value must be a string. Follow this format exactly:
            {
                "chain of thought": "Reasoning behind why a specific agent was chosen based on the user input.",
                "decision": "details_agent, order_taking_agent, or recommendation_agent. Pick one of those and only write the agent name.",
                "message": "Leave this empty."
            }
        """

        input_messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        input_messages += messages[-3:]
        
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        chatbot_output = check_json_output(self.client, self.model_name, chatbot_output)
        output = self.postprocess(chatbot_output)

        return output
    
    def postprocess(self, output):
        """
        Converts raw JSON string output into chatbot-compatible format
        with memory for routing to the correct agent.
        """
        clean_output = extract_json_block(output)
        output = json.loads(clean_output)

        dict_output = {
            "role": "assistant",
            "content": output['message'],
            "memory": {
                "agent": "classification_agent",
                "classification_decision": output["decision"]
            }
        }
        return dict_output