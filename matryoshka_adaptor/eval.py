import json
import logging
import os

import matplotlib.pyplot as plt
import mteb
import ptvsd
import torch
from model import (
    AdaptedSentenceTransformer,
    MatryoshkaAdaptor,
    SentenceTransformerTruncated,
)
from sentence_transformers import SentenceTransformer

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


def multiple_embedding_dimensionalities_eval(
    model,
    model_string,
    tasks=MTEB_BEIR_TASKS,
    encoding_batch_size=1200,
    embedding_dimensionalities=[16, 32, 64, 128, 192, 256, 320, 384],
) -> dict:
    # check if some file with results already exists
    eval_results = {}
    if os.path.exists(f"eval_{model_string}.json"):
        eval_results = json.load(open(f"eval_{model_string}.json"))

    logger.info(
        f"Eval {model_string} with {tasks} on {embedding_dimensionalities} truncations ..."
    )
    for dimensionality in embedding_dimensionalities:
        if str(dimensionality) not in eval_results:
            eval_results[str(dimensionality)] = {}

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

            # to align with original MTEB procedure
            if task == "MSMARCO":
                split = "dev"
            else:
                split = "test"

            results = evaluation.run(
                model_truncated,
                output_folder=f"results/{model_string}/{dimensionality}",
                eval_splits=[split],
                encode_kwargs={"batch_size": encoding_batch_size},
                overwrite_existing_files=True,
            )

            for task in results:
                eval_results[str(dimensionality)][task.task_name] = task.scores[split][
                    0
                ]["ndcg_at_10"]

            with open(f"eval_{model_string}.json", "w") as f:
                json.dump(eval_results, f, indent=4)

    logger.info(f"Eval results: {eval_results}")
    return eval_results


def plot_task_performances(eval_results, labels, tasks: list, figsize=(14, 4)):
    fig, axes = plt.subplots(1, len(tasks), figsize=figsize)
    colors = ["red", "blue", "green", "orange", "purple"]

    for task in tasks:
        if len(tasks) > 1:
            ax = axes[tasks.index(task)]
        else:
            ax = axes

        for i, eval_result in enumerate(eval_results):
            x_axis = []
            y_axis = []

            for dimensionality in eval_result:
                if task not in eval_result[dimensionality]:
                    continue

                x_axis.append(int(dimensionality))
                y_axis.append(eval_result[dimensionality][task])

                # annotate score
                ax.annotate(
                    f"{eval_result[dimensionality][task]:.3f}",
                    (int(dimensionality), eval_result[dimensionality][task]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

            # sort x and y
            x_axis, y_axis = zip(*sorted(zip(x_axis, y_axis)))

            ax.plot(x_axis, y_axis, color=colors[i], label=labels[i])

        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_axis, rotation=90)
        ax.set_title(task)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("Embedding Dimensionality")
        ax.set_ylabel("NDCG@10")
        ax.legend()

    fig.tight_layout()
    return fig


debug_mode = False

if debug_mode:
    print("Waiting for debugger to attach...")
    ptvsd.enable_attach(address=("localhost", 5678))
    ptvsd.wait_for_attach()


model_string = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim_full = 384
adaptors_to_evaluate = [
    "adaptor_supervised.pt",
    "adaptor.pt",
]
dimensionalities_to_evaluate = [16, 32, 64, 128, 192, 256, 320, 384]
tasks = ["HotpotQA"]

if __name__ == "__main__":
    logging.getLogger("mteb").setLevel(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformer(model_string)
    model.to(device)

    # eval for original model
    eval_results_original = multiple_embedding_dimensionalities_eval(
        model,
        "original",
        tasks=tasks,
        embedding_dimensionalities=dimensionalities_to_evaluate,
    )

    # eval for adapted models
    eval_results_adapted = []
    for adaptor_file in adaptors_to_evaluate:
        # load adapter from adaptor.pt
        adaptor = MatryoshkaAdaptor(embedding_dim_full)
        adaptor.load_state_dict(torch.load(adaptor_file))
        adaptor.eval()
        adaptor.to(device)
        model_adapted = AdaptedSentenceTransformer(model, adaptor, embedding_dim_full)

        # eval for adapted model
        eval_results_adapted.append(
            multiple_embedding_dimensionalities_eval(
                model_adapted,
                adaptor_file,
                tasks=tasks,
                embedding_dimensionalities=dimensionalities_to_evaluate,
            )
        )

    # plot results
    fig = plot_task_performances(
        [eval_results_original] + eval_results_adapted,
        ["original"] + adaptors_to_evaluate,
        tasks,
        figsize=(6, 4),
    )
    fig.savefig("eval_results.png", dpi=300)
