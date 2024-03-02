from dotenv import load_dotenv
import os

load_dotenv('api_keys.env')

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']