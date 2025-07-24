from dotenv import load_dotenv
import pandas as pd
import os
from .utils import get_chatbot_response, get_embedding, check_json_output
from openai import OpenAI
from copy import deepcopy
from pinecone import Pinecone
import json
load_dotenv()

class RecommendationAgent():
    def __init__(self, apriori_recommendation_path, popular_recommendation_path):
        # Initiazlied OpenAI client with environment-specified credentials
        self.client = OpenAI(
            api_key=os.getenv('RUNPOD_TOKEN'),
            base_url=os.getenv('RUNPOD_CHATBOT_URL')
        )
        self.model_name = os.getenv('MODEL_NAME')

        #Load apriori recommendation data from JSON
        with open(apriori_recommendation_path,  'r') as f:
            self.apriori_recommendations = json.load(f)

        # Load popularity-based recommendations from CSV
        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        self.products = self.popular_recommendations['product'].tolist()
        self.product_categories = list(set(self.popular_recommendations['product_category'].tolist()))

    def get_apriori_recommendation(self, products, top_k=5):
        """
        Returns top-k recommendations using the apriori algorithm data.
        Filters the predefined apriori mapping using input product list.
        """
        recommendation_list = []

        for product in products:
            if product in self.apriori_recommendations:
                recommendation_list += self.apriori_recommendations[product]
        
         # Sort by confidence score (descending) to get top recommendations
        recommendation_list = sorted(recommendation_list,key=lambda x: x['confidence'],reverse=True)
        
        recommendations = []
        recommendations_per_category = {}

        for recommendation in recommendation_list:
            # Limit 2 recommendations per category
            category = recommendation['product_category']
            product = recommendation['product']

            # Skip duplicates
            if product in recommendations:
                continue
            
            # Enforce 2 recommendations max per category
            if recommendations_per_category.get(category, 0) >= 2:
                continue

            recommendations.append(product)
            recommendations_per_category[category] = recommendations_per_category.get(category, 0) + 1

            # Make sure recommendations stay in top_k
            if len(recommendations) >= top_k:
                break

        return recommendations
    
    def get_popular_recommendation(self, product_categories=None, top_k=5):
        """
        Returns top-k popular products, optionally filtered by product category.
        Sorted by number of transactions (descending).
        """
        recommendation_df = self.popular_recommendations

        if type(product_categories) == str:
            product_categories = [product_categories]

        if product_categories is not None:
            recommendation_df = self.popular_recommendations[self.popular_recommendations['product_category'].isin(product_categories)]
        
        recommendation_df = recommendation_df.sort_values(by='num_of_transactions', ascending=False)

        if recommendation_df.shape[0] == 0:
            return []
        
        recommendations = recommendation_df['product'].tolist()[:top_k]
        
        return recommendations

    def recommendation_classification(self, message):
        """
        Uses LLM to classify the user's intent into one of:
        - apriori
        - popular
        - popular by category
        """
        system_prompt = """
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries. We have 3 types of recommendations:

        1. Apriori Recommendations: These are recommendations based on the user's order history. We recommend items that are frequently bought together with the items in the user's order.
        2. Popular Recommendations: These are recommendations based on the popularity of items in the coffee shop. We recommend items that are popular among customers.
        3. Popular Recommendations by Category: Here the user asks to recommend them product in a category. Like what coffee do you recommend me to get?. We recommend items that are popular in the category of the user's requested category.
        
        Here is the list of items in the coffee shop:
        """+ ",".join(self.products) + """
        Here is the list of Categories we have in the coffee shop:
        """ + ",".join(self.product_categories) + """

        Your task is to determine which type of recommendation to provide based on the user's message.

        Your output should be in a structured json format like so. Each key is a string and each value is a string. Make sure to follow the format exactly:
        {
        "chain of thought": "Write down your critical thinking about what type of recommendation is this input relevant to.",
        "recommendation_type": "'apriori' or 'popular' or 'popular by category'. Pick one of those and only write the word.",
        "parameters": "This is a python list. It's either a list of of items for apriori recommendations or a list of categories for popular by category recommendations. Leave it empty for popular recommendations. Make sure to use the exact strings from the list of items and categories above."
        }
        """

        input_messages = [{"role": "system", "content": system_prompt}] + message[-3:]

        chatbot_response = get_chatbot_response(self.client, self.model_name,input_messages)
        chatbot_output = check_json_output(self.client, self.model_name, chatbot_response)
        output = self.postprocess_classification(chatbot_response)

        return output
    
    def postprocess_classification(self, output):
        """
        Parses JSON response from the model to extract recommendation type and parameters.
        """
        output = json.loads(output)

        dict_output = {
            "recommendation_type": output['recommendation_type'],
            "parameters": output['parameters']
        }

        return output
    
    def get_recommendations_from_order(self, messages, order):
        """
        Handles generating a natural language response using LLM,
        based on apriori recommendations derived from a user's current order.
        """
        messages = deepcopy(messages)
        products = []

        for product in order:
            products.append(product['item'])

        recommendations = self.get_apriori_recommendation(products)
        recommendation_str = ", ".join(recommendations)

        system_prompt = """
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        your task is to recommend items to the user based on their input message. And respond in a friendly but concise way. Please be concise. And put it an unordered list with a very small description.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """

        prompt = f"""
        {messages[-1]['content']}
        Please recommend me those items  exactly: {recommendation_str}
        """

        messages[-1]['content'] = prompt
        input_message =[{'role': 'system', 'content': system_prompt}] + messages[-3:]

        chatbot_output = get_chatbot_response(self.client, self.model_name, input_message)
        output = self.postprocess(chatbot_output)

        return output
    

    def get_response(self, messages):
        """
        Entry point for generating recommendations.
        Classifies intent and delegates to the appropriate recommendation method.
        """
        messages = deepcopy(messages)

        recommendation_classifcation = self.recommendation_classification(messages)
        recommendation_type = recommendation_classifcation['recommendation_type']

        recommendation = []
        if recommendation_type == 'apriori':
            recommendation = self.get_apriori_recommendation(recommendation_classifcation['parameters'])
        elif recommendation_type == 'popular':
            recommendation = self.get_popular_recommendation()
        elif recommendation_type == 'popular by category':
            recommendation = self.get_popular_recommendation(recommendation_classifcation['parameters'])

        # Fallback response if no recommendation is available
        if not recommendation:
            return {'role': 'assistant', 'content': 'Sorry, I can\'t help with that. Can I help you with something else?'} 
        
        recommendation_str = ", ".join(recommendation)

        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        your task is to recommend items to the user based on their input message. And respond in a friendly but concise way. And put it an unordered list with a very small description.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend me those items exactly: {recommendation_str}
        """

        messages[-1]['content'] = prompt

        input_message = [{'role': 'system', 'content': prompt}] + messages[-3:]
    
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_message)
        output = self.postprocess(chatbot_output)

        return output
    
    def postprocess(self, output):
        """
        Standardizes the chatbot response format.
        """
        output = {
            'role': 'assistant',
            'content': output,
            'memory': {'agent': 'recommendation_agent'}
        }
        return output