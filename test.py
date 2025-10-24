import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, ScoredPoint, Filter
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://212.41.9.143:6333"

client = QdrantClient(url="http://212.41.9.143:6333", api_key="sk-4d2a1cbb2f8e4cba9a4a2b8cf1f2d3a2")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def search_sentences(query_text, collection="sentences2", top_k=5):
    query_vector = model.encode([query_text])[0].tolist()  # список чисел

    start_time = time.time()

    results = client.query_points(
        collection_name=collection,
        query=query_vector,  # Для последних версий нужно использовать именно этот параметр
        limit=10,
        using="fast-paraphrase-multilingual-minilm-l12-v2"

    ).points
    # print(query_vector)
    elapsed = time.time() - start_time
    print(f"Выполнено за {elapsed} seconds")
    for r in results:
        # print(r)
        print(f"- {r.payload['text']} (score={r.score})")

if __name__ == "__main__":
    while True:
        query = input("Введите фразу: ")
        if not query:
            break
        search_sentences(query)
