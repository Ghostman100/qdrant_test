import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, ScoredPoint, Filter
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://212.41.9.143:6333"

client = QdrantClient(url="http://212.41.9.143:6333")
model = SentenceTransformer("deepvk/USER-bge-m3")

def search_sentences(query_text, collection="sentences2", top_k=5):
    query_vector = model.encode([query_text])[0].tolist()  # список чисел

    start_time = time.time()

    results = client.query_points(
        collection_name=collection,
        query=query_vector,  # Для последних версий нужно использовать именно этот параметр
        limit=10,

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
