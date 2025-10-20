FROM python:3.11-slim

WORKDIR /app

# системные зависимости
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# копируем зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# создаём папку для NLTK
ENV NLTK_DATA=/usr/local/nltk_data

RUN mkdir -p $NLTK_DATA

# скачиваем нужный пакет один раз при сборке
RUN python -m nltk.downloader punkt -d $NLTK_DATA

# копируем код и книгу
COPY . .

# по умолчанию запускаем скрипт
CMD ["python", "book_to_qdrant_split.py"]
