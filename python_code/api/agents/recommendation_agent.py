from dotenv import load_dotenv
import pandas as pd
import os
from .utils import get_chatbot_response, get_embedding
from openai import OpenAI
from copy import deepcopy
from pinecone import Pinecone
import json
load_dotenv()

class RecommendationAgent():
    def __init__(self, apriori_recommendation_path, popular_recommendation_path):
        self.client = OpenAI(
            api_key=os.getenv('RUNPOD_TOKEN'),
            base_url=os.getenv('RUNPOD_CHATBOT_URL')
        )
        self.model_name = os.getenv('MODEL_NAME')

        with open(apriori_recommendation_path,  'r') as f:
            self.apriori_recommendations = json.load(f)

        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        self.products = self.popular_recommendations['products'].tolist()
        self.product_categories = self.popular_recommendations['product_category'].tolist()

    def get_popular_recommendation(self, product_categories=None, top_k=5):
        recommendation_df = self.popular_recommendations

        if type(product_categories) == str:
            product_categories = [product_categories]

        if product_categories is not None:
            recommendation_df = self.popular_recommendations[self.popular_recommendations['product_category'].isin(product_categories)]
        
        recommendation_df = recommendation_df.sort_values('number_of_transactions', ascending=False)

        if recommendation_df.shape[0] == 0:
            return []
        
        recommendations = recommendation_df['product'].tolist()[:top_k]
        
        return recommendations

