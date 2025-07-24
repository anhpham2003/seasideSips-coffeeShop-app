from typing import Protocol, List, Dict, Any

class AgentProtocol(Protocol):
    def get_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Defines the interface for agent classes.

        Parameters:
        - messages (List[Dict[str, Any]]): 
            A list of message dictionaries, each containing keys like 'role' and 'content',
            representing the dialogue history between the user and the assistant.

        Returns:
        - Dict[str, Any]: 
            A dictionary containing:
                - 'role': The role of the response (e.g., 'assistant')
                - 'content': The generated response message
                - 'memory': Additional metadata for routing or tracking (e.g., which agent responded)
        """