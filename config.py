"""Configuration settings for the Icebreaker Bot."""
import os
from dotenv import load_dotenv

load_dotenv()

# IBM watsonx.ai settings
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

url = "https://eu-gb.ml.cloud.ibm.com"
project_id = "d065193c-9ad1-468b-a895-50a5aed857a0"
api_key = os.environ.get("WATSONX_APIKEY")

credentials = {
    "url": url,
    "api_key": api_key,  # Using the api_key variable from environment variables
    "project_id": project_id  # Also including project_id for completeness
}
# Model settings
LLM_MODEL_ID = "ibm/granite-3-2b-instruct"
EMBEDDING_MODEL_ID = "ibm/slate-125m-english-rtrvr"

# ProxyCurl API settings
PROXYCURL_API_KEY = ""  # Replace with your API key

# Mock data URL
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1

# Node settings
CHUNK_SIZE = 500

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""