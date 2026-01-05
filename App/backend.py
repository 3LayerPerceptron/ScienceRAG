import os

def txt2bin(file_path):
    with open(file_path, "rb") as file:
        bin_data = file.read()
    return bin_data


def upload_dataset(rag_object, data_dir, chunk_method="naive", name="test_dataset", embedding_model="mistral-embed@Mistral"):
        
    dataset = rag_object.create_dataset(
        name=name,
        embedding_model=embedding_model,
        chunk_method=chunk_method
    )

    print(name)

    files = os.listdir(data_dir)

    document_list = [{"display_name" : file_name, "blob": txt2bin(os.path.join(data_dir, file_name))}for file_name in files]

    dataset.upload_documents(document_list)
    
    return dataset

def parse_documents(dataset):
    
    ids = [document.id for document in dataset.list_documents()]

    try:
        finished = dataset.parse_documents(ids)
        for doc_id, status, chunk_count, token_count in finished:
            print(f"Document {doc_id} parsing finished with status: {status}, chunks: {chunk_count}, tokens: {token_count}")
    except KeyboardInterrupt:
        print("\nParsing interrupted by user. All pending tasks have been cancelled.")
    except Exception as e:
        print(f"Parsing failed: {e}")

def generate():
    pass