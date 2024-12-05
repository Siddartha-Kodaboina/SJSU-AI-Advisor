
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://z44mbt93hez5cwy7.us-east-1.aws.endpoints.huggingface.cloud"
api_key = os.getenv("HF_API_KEY")
headers = {
	"Accept" : "application/json",
	"Content-Type": "application/json",
     "Authorization": f"Bearer {api_key}",
}



def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def llama2_7b_ysa(prompt: str) -> str:
    output = query({
        "inputs": prompt,
        "parameters": {}
    })

    print(output)
    response = output[0]['generated_text']

    return response

# %%time

prompt = "Who does YSA focus on helping?"
response = llama2_7b_ysa(prompt)

print(response)














# import requests

# API_URL = "https://z44mbt93hez5cwy7.us-east-1.aws.endpoints.huggingface.cloud"
# headers = {
#     "Authorization": f"Bearer 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",
#     "Content-Type": "application/json"
# }

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

# # Test the endpoint
# response = query({
#     "inputs": "Deploying my first endpoint was an amazing experience.",
#     "parameters": {
#         "max_new_tokens": 100,
#         "temperature": 0.7
#     }
# })
# print(response)