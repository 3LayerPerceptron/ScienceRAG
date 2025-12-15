import os
import requests



def test_upload_dataset(data_dir):

    files = [('files', (file, open(os.path.join(data_dir, file), 'rb'), 'text/plain')) for file in os.listdir(data_dir)]
    
    params = {
        'name': "test_API_small",
        'chunk_method': "naive",
        'embedding_model': "mistral-embed@Mistral"
    }

    response = requests.post(
        "http://localhost:8025/upload-dataset/",
        files=files,
        params=params
    )
    return response

def test_parse_documents(dataset_id):

    response = requests.post(
        "http://localhost:8025/parse-documents/",
        params={
            "dataset_id" : dataset_id
        }
    )

    return response


def test_retrieve(query, dataset_ids):
    
    response = requests.post(
        "http://localhost:8025/retrieve/",
        json={
            "query": query,
            "dataset_ids": dataset_ids,
            "limit": 3,
            "similarity_threshold": 0.2
        }
    )

    return response

def test_generate(query, dataset_ids):
    
    response = requests.post(
        "http://localhost:8025/generate/",
        json={
            "query": query,
            "dataset_ids": dataset_ids,
            "limit": 3,
            "similarity_threshold": 0.2,
            "model": "mistral-tiny"
        }
    )

    return response

def test_rag():
    pass

if __name__ == "__main__":
    print("#" * 80)
    print("\nUPLOAD DATASET TEST")

    data_dir = "./conv_smalls"
    response = test_upload_dataset(data_dir).json()
    print(response)

    print()
    print("#" * 80)
    print("\nPARSE DOCUMENTS TEST")

    dataset_id = response["dataset_id"]
    response = test_parse_documents(dataset_id).json()
    print(response)

    print()
    print("#" * 80)
    print("\nUPLOAD RETRIEVE TEST")
    response = test_retrieve("Is there such a thing as hierarchical time series?", dataset_ids=['aa4f5248d76f11f085d166c97ee06825']).json()
    print(response)

    print()
    print("#" * 80)
    print("\nUPLOAD RETRIEVE TEST")
    response = test_generate("Is there such a thing as hierarchical time series?", dataset_ids=['aa4f5248d76f11f085d166c97ee06825']).json()
    print(response)

    print()
    print("#" * 80)