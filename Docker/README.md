# Запуск докер контейнера

0. Необходимо поднять RAGFlow

1. После клонирования репозиотрия должен быть заполнен .env файл

2. Создание образа

```
sudo docker build -t science-rag-api .
```

3. Запуск контейнера

```
docker run -d -p 8025:8025 --network host --name science-rag science-rag-api
```

4. Теперь контейнер поднят, примеры запросов через curl и через request приведены в файлах request_examples.txt и test_api.py соответственно
