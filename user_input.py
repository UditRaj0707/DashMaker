# Process input files and build hybrid RAG index using Qdrant

import pandas as pd
from pathlib import Path
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from llama_cloud_services import LlamaParse
from qdrant_client import models
import os
from load_dotenv import load_dotenv
load_dotenv()

class UserInput:
    def __init__(self):
        self.embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_id)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.vector_store = None

    def process_files(self, file_paths):
        # self.loader = DoclingLoader(
        #     file_path=file_paths,
        #     export_type=ExportType.MARKDOWN,
        #     chunker=HybridChunker(tokenizer=self.embed_model_id),
        # )
        self.loader = LlamaParse(
            result_type="markdown",
            auto_mode=True,
            auto_mode_trigger_on_table_in_page=True,
            api_key=os.getenv("LLAMAPARSE_API_KEY")
        )
        docs = self.loader.get_json_result(file_paths)

        documents = [
            Document(
                page_content=page['md'],
                metadata={'id': f"doc_{k}_page_{i}", 'has_table': page['triggeredAutoMode'], 'file': doc['file_path']}
            ) 
            for k, doc in enumerate(docs)
            for i, page in enumerate(doc['pages'])
        ]

        if os.path.exists("./user_input_cache"):
            self.load_index()
            self.add_documents(documents)

        else:
            self.build_index(documents)
        

    def build_index(self, documents):
        qdrant_path = Path("./user_input_cache")
        qdrant_path.mkdir(exist_ok=True)

        self.vector_store = QdrantVectorStore.from_documents(
            documents,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            path="./user_input_cache",
            collection_name="user_input",
            retrieval_mode=RetrievalMode.HYBRID
        )
    
    def load_index(self):
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            path="./user_input_cache",
            collection_name="user_input",
            retrieval_mode=RetrievalMode.HYBRID
        )
    
    def add_documents(self, documents):
        print("Indexing...")
        self.vector_store.add_documents(documents)

    def search(self, query, k=3, filter=None):
        if filter:
            apply_filter = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key=f"metadata.has_table",
                                    match=models.MatchValue(
                                        value = True
                                    ),
                                ),
                            ]
                        )
            return self.vector_store.similarity_search(query, k=k, filter=apply_filter)
        else:
            return self.vector_store.similarity_search(query, k=k)


def test_parse():
    from llama_cloud_services import LlamaParse
    import json

    file_path = ["data/form-10k-exp.pdf"]

    parser = LlamaParse(
        result_type="markdown",
        auto_mode=True,
        auto_mode_trigger_on_table_in_page=True,
        api_key=os.getenv("LLAMAPARSE_API_KEY")
    )
    results = parser.get_json_result(file_path)
    print(len(results))
    with open("json_results_llama.json", "w", encoding='utf-8') as f:
        json.dump(results, f)
    

if __name__ == "__main__":
    user_input = UserInput()
    user_input.load_index()
    inputs = 'data/form-10k-exp.pdf'
    # user_input.process_files(inputs)
    results = user_input.search("  ", k=10, filter=True)
    print(len(results))
    for result in results:
        # print(result.page_content)
        print(result.metadata)
        print("\n")
    # test_parse()
    
    
