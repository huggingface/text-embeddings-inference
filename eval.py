import asyncio

import numpy as np

from mteb import MTEB
from aiohttp import ClientSession, ClientTimeout


async def request(sentence):
    r = {
        "inputs": sentence,
        "truncate": True
    }

    async with ClientSession(timeout=ClientTimeout(500*60)) as session:
        async with session.post('http://127.0.0.1:8080/embed', json=r) as resp:
            if resp.status != 200:
                raise RuntimeError(await resp.json())
            payload = await resp.json()

    return np.array(payload[0])


async def batch(sentences):
    return await asyncio.gather(*[request(sentence) for sentence in sentences])


class MyModel():
    def encode(self, sentences, batch_size=1024, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        results = []

        print(kwargs)

        # for i in range(len(sentences) // batch_size):
        #     results.extend(asyncio.run(batch(sentences[i * batch_size:(i + 1) * batch_size])))
        results = asyncio.run(batch(sentences))

        return results


model = MyModel()
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("llmrails/ember-v1", device="cuda")

evaluation = MTEB(tasks=["ArguAna"])
results = evaluation.run(model, output_folder=f"results/ember-candle-norm")
