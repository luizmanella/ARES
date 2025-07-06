from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Memries:
    """
    This is a new tech idea for memory storage. A Semantic RAG, per se.
    """
    def __init__(self):
        pass


def testing_doc_parsing():
    loader = PyPDFLoader("test_docs/test_ch1.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunked_docs = splitter.split_documents(docs)
    print(type(chunked_docs[0]), len(chunked_docs))
    print(chunked_docs[1])

# testing_doc_parsing()






import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# === Function: Compute BM25 scores and ranks ===
def get_bm25_scores(query_tokens, documents):
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query_tokens)
    ranks = np.argsort(-scores)  # descending
    rank_dict = {idx: rank + 1 for rank, idx in enumerate(ranks)}  # 1-based rank
    return scores, rank_dict

# === Function: Compute embedding scores and ranks ===
def get_embedding_scores(query, documents, model):
    doc_embeddings = model.encode(documents, convert_to_tensor=False)
    query_embedding = model.encode([query])[0]
    scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    ranks = np.argsort(-scores)
    rank_dict = {idx: rank + 1 for rank, idx in enumerate(ranks)}  # 1-based rank
    return scores, rank_dict

# === RRF scoring function ===
def rrf(rank, k=60):
    return 1 / (k + rank)

# === Function: Combine scores using RRF ===
def compute_rrf_fusion(bm25_ranks, embed_ranks, bm25_weight=0.25, embed_weight=1.0, k=60):
    scores = []
    for i in range(len(bm25_ranks)):
        bm25_rank = bm25_ranks.get(i, k)
        embed_rank = embed_ranks.get(i, k)
        bm25_rrf = bm25_weight * rrf(bm25_rank, k)
        embed_rrf = embed_weight * rrf(embed_rank, k)
        total = bm25_rrf + embed_rrf
        scores.append((i, bm25_rank, embed_rank, bm25_rrf, embed_rrf, total))
    return scores

# === Main Execution ===
documents = [
    "The Eiffel Tower is located in Paris and is one of the most famous landmarks in France.",
    "Machine learning models can identify patterns in data and make predictions.",
    "The Great Wall of China is a historic fortification stretching thousands of miles.",
    "Paris is known for its cafe culture and historical monuments like Notre-Dame.",
    "Supervised learning uses labeled data to train machine learning models.",
    "Beijing is the capital of China and a major cultural and political center.",
]

query = "Where is the Eiffel Tower located?"
query_tokens = query.lower().split()

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get scores and ranks
bm25_scores, bm25_rank_dict = get_bm25_scores(query_tokens, documents)
embed_scores, embed_rank_dict = get_embedding_scores(query, documents, model)

# Compute fused RRF scores
rrf_results = compute_rrf_fusion(bm25_rank_dict, embed_rank_dict)

# Format output
df = pd.DataFrame(rrf_results, columns=[
    "Doc ID", "BM25 Rank", "Embed Rank", "BM25 RRF", "Embed RRF", "RRF Total"
])
df["Document"] = [documents[i] for i in df["Doc ID"]]
df = df.sort_values("RRF Total", ascending=False)

# Display
print(df[["Document", "BM25 Rank", "Embed Rank", "BM25 RRF", "Embed RRF", "RRF Total"]].to_string(index=False))
