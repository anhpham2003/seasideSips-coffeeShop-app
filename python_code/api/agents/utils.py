import re

def get_chatbot_response(client, model_name, messages, temperature=0.0):
    input_messages = []

    for message in messages:
        input_messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=0.0,
        top_p=0.8,
        max_tokens=2000
    ).choices[0].message.content
    return response

def get_embedding(embedding_client, model_name, text_input):
    output = embedding_client.embeddings.create(input=text_input, model=model_name)

    embeddings = []
    for embedding_ocject in output.data:
      embeddings.append(embedding_ocject.embedding)

    return embeddings

def check_json_output(client, model_name, json_string):
    prompt = f"""
        You will check this JSON string and correct any mistakes that would make it invalid. Then you will return ONLY the corrected JSON string. 
        If it's already correct, just return it directly.
        If there is any text before or after the json string, remove it.
        Do NOT return a single letter outside of the json string.
        The first thing you write should be open curly brace of the json and the last thing you write should be the closing curly brace.

        You should check the json string for the following text between triple backticks
        ```
        {json_string}
        ```
    """

    messages = [{'role': 'user', 'content': prompt}]
    response = get_chatbot_response(client, model_name, messages)
    response = response.replace("```","")

    return response

def extract_json_block(text: str) -> str:
    """
    Make sure that the json string is extracted correctly
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else ""