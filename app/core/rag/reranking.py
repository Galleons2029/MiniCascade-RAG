from app.core.config import settings
import requests
import json
import os

import app.core.logger_utils as logger_utils

logger = logger_utils.get_logger(__name__)
# from xinference.client import Client

# client = Client("http://localhost:9997")
# model = client.get_model('bge-reranker-v2-m3')


class Reranker:
    @staticmethod
    def generate_response(query: str, passages: list[str], keep_top_k: int) -> list[str]:
        payload = {
            "model": settings.Silicon_model_rerank,
            "query": query,
            "documents": passages,
            "top_n": keep_top_k,
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80,
        }

        headers = {"Authorization": f"Bearer {settings.Silicon_api_key3}", "Content-Type": "application/json"}

        response = requests.post(os.path.join(settings.Silicon_base_url, "rerank"), json=payload, headers=headers)
        response_data = json.loads(response.text)

        logger.debug(response_data)
        ranked_indices = [result["index"] for result in response_data["results"]]
        reranked_passages = [passages[idx] for idx in ranked_indices]
        # reranked_passages =  model.rerank(passages, query, return_documents=True)

        return [i["document"]["text"] for i in reranked_passages["results"]]

        # return reranked_passages


if __name__ == "__main__":
    # from xinference.client import Client
    from FlagEmbedding import FlagReranker

    # Setting use_fp16 to True speeds up computation with a slight performance degradation (if using gpu)
    reranker = FlagReranker(
        "/root/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/12e974610ba9083ed95f3edf08d7e899581f4de4",
        devices=["cuda:2"],
        use_fp16=True,
    )
    # client = Client("http://localhost:9997")

    # model = client.get_model('bge-reranker-v2-m3')

    query = "A man is eating pasta."
    passages = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
    ]
    # response = model.rerank(passages, query, return_documents=True)
    # ans = [ i['document']['text'] for i in response['results']]
    # print(ans)

    # start_time = time.time()
    # for i in range(10):
    #     model.rerank(passages, query)
    # stage1 = time.time() - start_time
    # print(stage1)

    # keep_top_k = 4
    # for i in range(10):
    #     rerank_passages = Reranker.generate_response(query, passages, keep_top_k)
    #     print(rerank_passages)

    # stage2 = time.time() - start_time - stage1
    # print(stage2)
    # for i in range(10):
    #     score = reranker.compute_score(['query', 'passage'])
    # # or set "normalize=True" to apply a sigmoid function to the score for 0-1 range
    # #score = reranker.compute_score(['query', 'passage'], normalize=True)
    #
    #     #print(score)
    # stage3 = time.time() - start_time - stage2
    # print(stage3)
