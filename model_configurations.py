import os
from dotenv import load_dotenv
load_dotenv()

configurations = {
    "text-embedding-ada-002": {
        "api_base": os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
        "api_key": os.getenv('AZURE_OPENAI_EMBEDDING_KEY'),
        "deployment_name": os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDING'),
        "api_version": os.getenv('AZURE_OPENAI_VERSION'),
        "model": os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDING_MODEL_NAME'),
        "openai_type": os.getenv('AZURE_OPENAI_TYPE')
    }
}

def get_model_configuration(model_version):
    return configurations.get(model_version)
