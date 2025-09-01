import json
import uuid
from app.core import logger_utils
from reasoning import ReasoningPipeline
from collections import defaultdict
from qdrant_client import QdrantClient, models
from app.pipeline.feature_pipeline.models.raw import DocumentRawModel
from app.core.config import settings
from app.core.mq import publish_to_rabbitmq

# 设置日志记录器
logger = logger_utils.get_logger(__name__)

def read_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def batch_inference(input_file, output_file="reasoning_task1_output.jsonl"):
    test_data = read_jsonl(input_file)
    inference_endpoint = ReasoningPipeline(mock=False)

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(test_data):

            interaction_id = item.get("interaction_id", "")
            question_type = item.get("question_type", "")
            q_text = item.get("query", "")
            gt_list = item.get("alt_ans", [])
            search_results = item.get("search_results", [])

            if not q_text or not search_results:
                continue

            # --- 每条测试数据单独创建 Qdrant 知识库 ---
            collection_name = f"zsk_test_{idx}"
            client = QdrantClient(url="http://localhost:6333")

            # 如果集合已存在则删除
            existing_collections = [col.name for col in client.get_collections().collections]
            if collection_name in existing_collections:
                client.delete_collection(collection_name=collection_name)

            # 创建集合
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=settings.EMBEDDING_SIZE, distance=models.Distance.COSINE),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99,
                                                           always_ram=True, ), ),
            )

            # 将 search_results 插入知识库
            for result in search_results:
                content = result['page_snippet']
                doc_entry = DocumentRawModel(
                    knowledge_id=collection_name,
                    doc_id=str(uuid.uuid4()),
                    path='file',
                    filename='file',
                    content=content,
                    type="documents",
                    entry_id=str(uuid.uuid4())
                ).model_dump_json()
                # 发送到 RabbitMQ 进行处理/写入 Qdrant
                publish_to_rabbitmq(queue_name='test_files', data=doc_entry)

            # 推理时使用本条数据对应的 doc_names
            doc_names = [result['page_snippet'] for result in search_results]

            response = inference_endpoint.generate(
                query=q_text,
                enable_rag=True,
                sample_for_evaluation=True,
                doc_names=[f"zsk_test_{idx}"]
            )

            # 保存结果
            result_json = {
                "interaction_id": interaction_id,
                "question_type": question_type,
                "question": q_text,
                "answer": response['answer'],
                "ground_truths": gt_list,
                "contexts": response['context']
            }

            logger.info(f"问题：{q_text}")
            logger.info(f"回答：{response['answer']}")
            logger.info("=" * 50)

            json.dump(result_json, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    input_file = "/home/liujialong/project/MiniCascade-RAG/test/evaluation/rag_evaluate/CRAG/task1/crag_task_1_dev_v4_release.jsonl"
    output_file = "/home/liujialong/project/MiniCascade-RAG/test/evaluation/rag_evaluate/results/reasoning_task1_output.jsonl"
    batch_inference(input_file, output_file)
