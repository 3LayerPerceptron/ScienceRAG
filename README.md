## Подготовка датасета Open RAG Benchmark для RAG

В ноутбуке `01_prepare_open_ragbench.ipynb` скачиваются файлы датасета Open RAG Benchmark c Huggingface (https://huggingface.co/datasets/vectara/open_ragbench) и кладутся в папку `data\raw`.

Тексты статей разбиваются на чанки специальной функцией и все кладутся в `data\processed\chunks.jsonl`.
Этот файл пушится на HuggingfaceHub (https://huggingface.co/datasets/Ilya-huggingface/open_ragbench_chunks)

Примеры можно посмотреть в `data\samples\chunks_sample.jsonl`.
Сюда (на Гитхаб) мы все файлы не подгружаем, поскольку они объёмные.



