import requests
from dotenv import load_dotenv
import base64
import os
import time

load_dotenv()

DID_KEY = os.getenv("D_ID_KEY")


# Encode the "username:apikey" string to Base64
did_key_encoded = base64.b64encode(DID_KEY.encode("utf-8")).decode("utf-8")

#  function to generate the  video by the D-ID
def generate_video_from_did(image_url=None, text="Making videos is easy with D-ID", agent_name=None):
    url = "https://api.d-id.com/talks"

    payload = {
        "source_url": "https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg",
        "script": {
            "type": "text",
            "subtitles": "false",
            "provider": {
                "type": "microsoft",
                "voice_id": "en-US-JennyNeural"
            },
            "input": text,
            "ssml": "false"
        },
        "config": { "fluent": "false" }
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Basic {did_key_encoded}"
    }

    response = requests.post(url, json=payload, headers=headers)

    return response.json()


#  fetch the video from the D-ID
def fetch_video_from_did(video_id):
    while True:
        url = f"https://api.d-id.com/talks/{video_id}"

        print("Fetching video from URL:", url)

        headers = {
            "accept": "application/json",
            "authorization": f"Basic {did_key_encoded}"
        }

        response = requests.get(url, headers=headers)
        print("Response status code:", response.json())
        
        #  if result url in response then return the result url
        if response.json().get("result_url"):
            print("Video processing completed.")
            return response.json()

#  funtion to create the animation of the  agent
def  create_animation(sourece_url=None, webhook_url=None):
    print('animation creation is strated')
    url = "https://api.d-id.com/animations"
    if not sourece_url:
        sourece_url="https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg"

    payload = { 
                "source_url": sourece_url ,
                "face": {
                        "top_left": [512, 512],
                        "detection": {
                            "top": 800,
                            "left": 800,
                            "bottom": 800,
                            "right": 800
                        },
                        "size": 512
                    },
                  "config": {
                    "stitch": True,
                },
                "webhook": webhook_url,
                "driver_url": "bank://subtle/driver-15"               
                }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization":  f"Basic {did_key_encoded}"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)

    res=response.json()

    return res.get('id')


#  function to get the presenter list
def get_presenter_list():
    """
    Fetch the list of presenters from the D-ID API.
    """
    try:
        url = "https://api.d-id.com/clips/presenters?limit=2000"

        headers = {
            "accept": "application/json",
            "authorization": f"Basic {did_key_encoded}"
        }

        response = requests.get(url, headers=headers)
        data=response.json()

        # #  only store the data in this train id is not used
        filtered_data=[item for item in data['presenters'] if "train_id" not in item.keys() and item.get("is_greenscreen", False)== False and item.get("idle_video", False)]
        return filtered_data
    
    except Exception as e:
        print("Error fetching presenter list:", e)
        return []
    
#  get the voice_id
def get_voice_list(gender: str='male', language: str='english'):
    """
    Fetch the list of voices from the D-ID API.
    """
    try:
        url = "https://api.d-id.com/tts/voices?provider=microsoft"
        

        headers = {
            "accept": "application/json",
            "authorization": f"Basic {did_key_encoded} "
        }

        response = requests.get(url, headers=headers)
        data = response.json()

        filtered_voices = [
            voice for voice in data
            if voice.get("gender", "").lower() == gender
            and language in voice.get("languages", [{}])[0].get("language", "").lower()
        ]

        return filtered_voices

    except Exception as e:
        print("Error fetching voice list:", e)
        return []