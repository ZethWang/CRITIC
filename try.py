from src.tools.perspective_api import PerspectiveAPI
from dotenv import find_dotenv,load_dotenv
import os
import httplib2
import socket



# _ = load_dotenv(find_dotenv())
# PERSPECTIVE_API_KEY=os.getenv('PERSPECTIVE_API_KEY')
# api = PerspectiveAPI(api_key=PERSPECTIVE_API_KEY)
from googleapiclient import discovery
import json

API_KEY = 'AIzaSyAzlvKwZYpJPw4DbP8ht5EjsRrvWs-4KNU'

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
