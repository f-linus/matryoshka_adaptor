import ptvsd
from sentence_transformers import SentenceTransformer

print("Waiting for debugger to attach...")
ptvsd.enable_attach(address=("localhost", 5678))
ptvsd.wait_for_attach()

sentences = ["This is an example sentence", "Each sentence is converted"]

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    print(embeddings)
