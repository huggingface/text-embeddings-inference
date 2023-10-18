from text_embeddings_inference_client import Client
from text_embeddings_inference_client.models import MyDataModel
from text_embeddings_inference_client.api.my_tag import get_my_data_model
from text_embeddings_inference_client.types import Response

client = Client(base_url="https://api.example.com")

with client as client:
    my_data: MyDataModel = get_my_data_model.sync(client=client)
    # or if you need more info (e.g. status_code)
    response: Response[MyDataModel] = get_my_data_model.sync_detailed(client=client)
