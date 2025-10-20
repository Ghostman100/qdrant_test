import re
import time
import os
import uuid

import nltk
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance



# --- Настройки ---
BOOK_PATH = "v_and_m_1-2.txt"
BOOK_PATH_2 = "v_and_m_3-4.txt"
BOOK_PATH_3 = "sherlock_dog.txt"
# FOLDER_PATH = 'books'

QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "deepvk/USER-bge-m3"

SHORT_COLLECTION = "short_fragments"
LONG_COLLECTION = "sentences"

BATCH_SIZE = 128

# --- Инициализация ---
nltk.download('punkt_tab')

time.sleep(3)
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

VECTOR_SIZE = model.get_sentence_embedding_dimension()


def ensure_collection(name: str):
    """Создаёт коллекцию, если её нет."""
    collections = [c.name for c in client.get_collections().collections]
    if name not in collections:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def split_text(text: str):
    """Разделяет текст на короткие фразы и предложения."""
    # очищаем
    text = re.sub(r"\s+", " ", text.strip())

    # короткие записи (1–3 слова)
    short_fragments = re.findall(r"\b\w+(?:\s+\w+){0,2}\b", text)

    # длинные записи — предложения
    sentences = nltk.sent_tokenize(text)

    return short_fragments, sentences


def embed_and_store(collection_name: str, texts: list[str], type_label: str):
    """Векторизует и записывает в Qdrant батчами."""
    ensure_collection(collection_name)

    print(f"📦 Загрузка в коллекцию '{collection_name}' ({len(texts)} элементов)")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Векторизация {type_label}"):
        batch = texts[i : i + BATCH_SIZE]
        vectors = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)

        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": str(uuid.uuid4()),
                    "vector": vec.tolist(),
                    "payload": {"text": batch[j], "type": type_label},
                }
                for j, vec in enumerate(vectors)
            ],
        )


def process_book(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    short_frags, sentences = split_text(text)

    print(f"📘 Книга загружена")
    print(f"— коротких фрагментов: {len(short_frags)}")
    print(f"— предложений: {len(sentences)}")

    # embed_and_store(SHORT_COLLECTION, short_frags, "short")
    embed_and_store(LONG_COLLECTION, sentences, "sentence")

    print("✅ Всё сохранено в Qdrant!")


if __name__ == "__main__":
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(FOLDER_PATH, filename)
            try:
                process_book(file_path)
            except:
                print('error asdasd', filename)


