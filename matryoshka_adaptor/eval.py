import json
import logging
import os

import mteb

MTEB_BEIR_TASKS = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackEnglishRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "Robust04InstructionRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class SentenceTransformerTruncated:
    def __init__(self, model, embedding_length):
        self.model = model
        self.prompts = model.prompts  # unfortunately this is necessary
        self.embedding_length = embedding_length

    def encode(self, sentences, **kwargs):
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, : self.embedding_length]


def multiple_embedding_dimensionalities_eval(
    model,
    model_string,
    tasks=MTEB_BEIR_TASKS,
    full_embedding_dimensionality=384,
    dimensionality_reduction_steps=50,
) -> dict:
    embedding_dimensionalities = range(
        full_embedding_dimensionality, 0, -dimensionality_reduction_steps
    )

    # check if some file with results already exists
    eval_results = {}
    if os.path.exists(f"{model_string}_eval.json"):
        eval_results = json.load(open(f"{model_string}_eval.json"))

    logger.info(
        f"Eval {model_string} with {tasks} on {embedding_dimensionalities} truncations ..."
    )
    for dimensionality in embedding_dimensionalities:
        if dimensionality not in eval_results:
            eval_results[dimensionality] = {}

        model_truncated = SentenceTransformerTruncated(model, dimensionality)

        # only do tasks not done yet
        for task in tasks:
            if task in eval_results[str(dimensionality)]:
                logger.info(
                    f"Task {task} already evaluated for dimensionality {dimensionality}, skipping ..."
                )
                continue

            mteb_tasks = mteb.get_tasks(tasks=[task])

            evaluation = mteb.MTEB(tasks=mteb_tasks)
            results = evaluation.run(
                model_truncated,
                output_folder=f"results/{model_string}/{dimensionality}",
                eval_splits=["test"],
            )

            for task in results:
                eval_results[str(dimensionality)][task.task_name] = task.scores["test"][
                    0
                ]["ndcg_at_10"]

            with open(f"{model_string}_eval.json", "w") as f:
                json.dump(eval_results, f)

    logger.info(f"Eval results: {eval_results}")
    return eval_results
