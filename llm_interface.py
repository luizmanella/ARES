from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
from os import environ as env
import json

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
DB_PATH = env.get("DATABASE_PATH")


class LLMInterface:
    """
    A class designed to handle working with the LLM, abstracting its 
    complexities and making it easy for the different Cognitive Modules
    to use.
    """
    def __init__(self, model, max_tokens, temperature=0) -> None:
        self.client = OpenAI(api_key = env.get("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature



    def query(self, messages, json_schema=None):

        if json_schema:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    'type':'json_schema',
                    'json_schema': json_schema
                },
                temperature=self.temperature
            )
            return json.loads(response.choices[0].message.content)
        else:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
