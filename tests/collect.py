
import requests
import torch
import argparse
import json
import os

parser = argparse.ArgumentParser(description='Assets collection')
parser.add_argument('--model-id', help='Model id', required=True)
parser.add_argument('--n_inp', help='Number of inputs', required=True, type=int)
parser.add_argument('--flash', action='store_true')

args = parser.parse_args()

url = f"http://0.0.0.0:80/embed"

INPUTS = [
    "What is Deep Learning?",
    "Today I am in Paris and I would like to",
    "Paris weather is",
    "Great job"
]

data = {"inputs": INPUTS[:args.n_inp]}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)

embedding = torch.Tensor(json.loads(response.text))

postfix = ""
if not args.flash:
    postfix = "_no_flash"

save_path = f"./assets/{args.model_id.replace('/', '-')}_inp{args.n_inp}{postfix}.pt"
print(f"Saving embedding of shape {embedding.shape} to {save_path}")
torch.save(embedding, save_path)