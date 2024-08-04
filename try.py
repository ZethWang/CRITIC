from src.tools.perspective_api import PerspectiveAPI
from dotenv import find_dotenv,load_dotenv
import os
import httplib2
import socket
from dotenv import find_dotenv,load_dotenv
_ = load_dotenv(find_dotenv())



# _ = load_dotenv(find_dotenv())
# PERSPECTIVE_API_KEY=os.getenv('PERSPECTIVE_API_KEY')
# api = PerspectiveAPI(api_key=PERSPECTIVE_API_KEY)
from googleapiclient import discovery
import json

API_KEY = os.getenv('ERSPECTIVE_API_KEY')

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

analyze_request = {
  'comment': { 'text': 'friendly greetings from python' },
  'requestedAttributes': {'TOXICITY': {}}
}

response = client.comments().analyze(body=analyze_request).execute()
print(json.dumps(response, indent=2))
