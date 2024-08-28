import logging

import ptvsd
from sentence_transformers import SentenceTransformer

from matryoshka_adaptor.eval import multiple_embedding_dimensionalities_eval

debug_mode = False
model_string = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim_full = 384


if debug_mode:
    print("Waiting for debugger to attach...")
    ptvsd.enable_attach(address=("localhost", 5678))
    ptvsd.wait_for_attach()


if __name__ == "__main__":
    logging.getLogger("mteb").setLevel(logging.INFO)

    model = SentenceTransformer(model_string)
    eval_results = multiple_embedding_dimensionalities_eval(
        model, "base", tasks=["SciFact", "ArguAna", "Touche2020"]
    )

    print(eval_results)
