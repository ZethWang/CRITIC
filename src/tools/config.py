import yaml
from pathlib import Path
from dotenv import find_dotenv,load_dotenv
import os
## TODO: upload Search API
_ =load_dotenv(find_dotenv())


## Perspective API
PERSPECTIVE_API_KEY = os.getenv('PERSPECTIVE_API_KEY')

PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
