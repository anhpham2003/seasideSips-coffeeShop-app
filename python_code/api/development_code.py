from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    AgentProtocol,
                    RecommendationAgent
                    )
import os
from typing import Dict
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()


def main():
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()

    agent_dict: Dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "recommendation_agent": RecommendationAgent(
                                    os.path.join(folder_path, "recommendation_objects/apriori_recommendation.json"),
                                    os.path.join(folder_path, "recommendation_objects/popularity_recommendation.csv")
                                )
    }

    messages = []

    while True:
        # os.system('cls' if os.name == 'nt' else 'clear')

        print("\n\n Print Messages ...........")
        for message in messages:
            print(f"{message['role']}: {message['content']}")

            # Get user input
        prompt = input('User: ')
        messages.append({'role': 'User', 'content': prompt})

        # Get Guard Agent's Response
        guard_agent_response = guard_agent.get_response(messages)
        print("GUARD AGENT OUTPUT:", guard_agent_response)

        if guard_agent_response['memory']['guard_decision'] == 'not allowed':
            messages.append(guard_agent_response)
            continue

        # Get classification agent's response
        classification_agent_repsonse = classification_agent.get_response(messages)
        chosen_agent = classification_agent_repsonse['memory']['classification_decision']
        print("Chosen Agent: ", chosen_agent)

        # Get the chosen agent's repsonse
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)

        messages.append(response)

if __name__ == "__main__":
    main()