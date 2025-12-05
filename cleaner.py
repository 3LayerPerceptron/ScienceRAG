import os
import json

from tqdm.auto import tqdm

def main(dir_path):
    for file in tqdm(os.listdir(dir_path), total=1_000):
        with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as json_file:
            
            # file.dict_keys(['title', 'sections', 'id', 'authors', 'categories', 'abstract', 'updated', 'published'])
            data = json.load(json_file)
        converted_path = os.path.join("/".join(dir_path.split("/")[:-1]), "converted")
        os.makedirs(converted_path, exist_ok=True)

        with open(f"{os.path.join(converted_path, file[:-5])}.txt", "w") as output_file:
            output_file.write(f"{data['title']}\n\n")
            output_file.write(f"{data['abstract']}\n\n")
            for section in data["sections"][:-1]:
                
                #section.dict_keys(['section_id', 'text', 'tables', 'images'])
                output_file.write(f"{section['text']}\n\n")

if __name__ == "__main__":
    main("./open_ragbench/pdf/arxiv/corpus")