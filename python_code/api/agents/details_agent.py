from dotenv import load_dotenv
import os
from .utils import get_chatbot_response, get_embedding
from openai import OpenAI
from copy import deepcopy
from pinecone import Pinecone
load_dotenv()

class DetailsAgent():
    def __init__(self):
        # Initialize OpenAI chat client for generating responses
        self.client = OpenAI(
            api_key = os.getenv('RUNPOD_TOKEN'),
            base_url = os.getenv('RUNPOD_CHATBOT_URL')
        )
        self.model_name = os.getenv('MODEL_NAME')

        # Initialize OpenAI client for embeddings
        self.embedding_client = OpenAI(
            api_key = os.getenv('RUNPOD_TOKEN'),
            base_url=os.getenv('RUNPOD_EMBEDDING_URL')
        )

        # Initialize Pinecone vector database client
        self.pc =  Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = os.getenv('PINECONE_INDEX_NAME')

    def get_closest_results(self, index_name, input_embeddings, top_k=2):
        """
        Queries Pinecone index using user input embeddings to retrieve top_k similar entries.
        Only returns metadata (e.g., text) for use in the LLM prompt.
        """
        index = self.pc.Index(index_name)
        
        results = index.query(
            namespace='ns1',
            vector=input_embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        return results
    
    def get_response(self, messages):
        """
        Responds to user input by:
        1. Extracting the user query
        2. Getting relevant context from Pinecone
        3. Feeding both into a prompt for the LLM
        4. Returning the LLM-generated response
        """
        messages = deepcopy(messages)

        user_message = messages[-1]['content']
        # Generate embedding for user message
        embeddings = get_embedding(self.embedding_client, self.model_name, user_message)[0]
        #  Retrieve relevant context using Pinecone
        result = self.get_closest_results(self.index_name, embeddings)
        # Combine matched knowledge into a context string
        source_knowledge = "\n".join([x['metadata']['text'].strip() + '\n' for x in result['matches']])

        # Prompt that includes context + query
        prompt = f"""
            Using the contexts below answer the query:
                Contexts:
                {source_knowledge}

                Query: {user_message}
        """

        system_prompt = """
            You are a customer support agent for a coffee shop called Seaside Sips. You should answer every questions as if you a waiter and provide the necessary information to the user regarding their orders.
        """
        messages[-1]['content'] = prompt

        # Build input message stack (system + last 3 turns)
        input_message = [{"role": "system", "content": system_prompt}] + messages[-3:]
        
        # Get final response from LLM
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_message)
        output = self.postprocess(chatbot_output)

        return output
    
    def postprocess(self, output):
        """
        Formats the raw model output into a structured response with agent memory.
        """
        output = {
            "role": "assistant",
            "content": output,
            "memory": {
                "agent": "details_agent"
            }
        }
        
        return output
