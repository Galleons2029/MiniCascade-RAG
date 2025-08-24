import argparse

from config import settings
from app.core.logger_utils import get_logger
from app.core.opik_utils import create_dataset_from_artifacts
from llm_twin import LLMTwin
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, LevenshteinRatio, Moderation

from .style import Style

logger = get_logger(__name__)


def evaluation_task(x: dict) -> dict:
    inference_pipeline = LLMTwin(mock=False)
    result = inference_pipeline.generate(
        query=x["instruction"],
        enable_rag=False,
    )
    answer = result["answer"]

    return {
        "input": x["instruction"],
        "output": answer,
        "expected_output": x["content"],
        "reference": x["content"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate monitoring script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="LLMTwinMonitoringDataset",
        help="Name of the dataset to evaluate",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name

    logger.info(f"Evaluating Opik dataset: '{dataset_name}'")

    dataset = create_dataset_from_artifacts(
        dataset_name="LLMTwinArtifactTestDataset",
        artifact_names=[
            "articles-instruct-dataset",
            "repositories-instruct-dataset",
        ],
    )
    if dataset is None:
        logger.error("Dataset can't be created. Exiting.")
        exit(1)

    experiment_config = {
        "model_id": settings.MODEL_ID,
    }
    scoring_metrics = [
        LevenshteinRatio(),
        Hallucination(),
        Moderation(),
        Style(),
    ]
    evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=scoring_metrics,
        experiment_config=experiment_config,
    )


if __name__ == "__main__":
    main()
