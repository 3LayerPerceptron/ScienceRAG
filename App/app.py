import os
import shutil
import tempfile
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from ragflow_sdk import RAGFlow
from fastapi import FastAPI, File, UploadFile

from backend import upload_dataset, parse_documents

class RetrieveRequest(BaseModel):
    query: str
    dataset_ids: List[str]
    limit: int = 10
    similarity_threshold: float = 0.2

class GenerateRequest(RetrieveRequest):
    model: str

rag_object = None
llm_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_object
    global llm_client
    
    rag_object = RAGFlow(
        api_key=os.getenv("RAGFLOW_API_KEY"), 
        base_url=os.getenv("RAGFLOW_BASE_URL")
    )

    llm_client = OpenAI( 
        base_url=os.getenv("LLM_BASE_URL"),
    )

    yield

app = FastAPI(
    title="ScienceRAG_API", 
    lifespan=lifespan
)

def txt2bin(file_path):
    with open(file_path, "rb") as file:
        return file.read()

@app.post("/upload-dataset/")
async def upload_dataset_endpoint(
    files: list[UploadFile] = File(...),
    name: str = "test_dataset",
    chunk_method: str = "naive",
    embedding_model: str = "mistral-embed@Mistral"
):

    temp_dir = tempfile.mkdtemp()
    
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
        
        dataset = upload_dataset(
            rag_object=rag_object,
            data_dir=temp_dir,
            chunk_method=chunk_method,
            name=name,
            embedding_model=embedding_model
        )
        
        return {
            "status": "success",
            "dataset_id": dataset.id
        }
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/parse-documents/")
async def parse_documents_endpoint(dataset_id: str):

    parse_documents(rag_object.list_datasets(id=dataset_id)[0])
    
    return {"status": "parsing_success", "dataset_id": dataset_id}


@app.post("/retrieve/")
async def retrieve_endpoint(request: RetrieveRequest):
    result = rag_object.retrieve(
        question=request.query,
        dataset_ids=request.dataset_ids,
        top_k=request.limit,
        similarity_threshold=request.similarity_threshold
    )
    return result

def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n".join([
        f"Document {i+1}:\n{chunk.content[:500]}..."
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    prompt = f"""You are an assistant that have access to multiple documents.

            Context retrieved from the documents:
            {context}

            User's query: {query}

            Give an answer based on retrieved context. If context does not contain the answer say so"""
    
    return prompt

@app.post("/generate/")
async def generate_answer(request: GenerateRequest):
    chunks = rag_object.retrieve(
        question=request.query,
        dataset_ids=request.dataset_ids,
        top_k=request.limit,
        similarity_threshold=request.similarity_threshold
    )
    print([chunk.document_id for chunk in chunks])
    sources = [rag_object.list_datasets(id=request.dataset_ids[0])[0].list_documents((chunk.document_id))[0].name for chunk in chunks]

    prompt = build_rag_prompt(request.query, chunks)
    
    response = llm_client.chat.completions.create(
        model=request.model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "prompt_used": prompt[:500] + "..."
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "science_rag_api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8025)