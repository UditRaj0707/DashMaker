from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from scrap_doc import DocumentationScraper
from pathlib import Path
import os
from load_dotenv import load_dotenv
load_dotenv()


class VectorIndexer:
    def __init__(self):
        self.vector_store = None
        # self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # self.embeddings = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    def index_docs(self, docs):
        qdrant_path = Path("./qdrant_db")
        qdrant_path.mkdir(exist_ok=True)

        self.vector_store =  QdrantVectorStore.from_documents(
            docs,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            path="./documentation_cache",
            collection_name="documentation",
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def load_index(self):
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            path="./documentation_cache",
            collection_name="documentation",
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def search(self, query):
        # query = "What did the president say about Ketanji Brown Jackson"
        found_docs = self.vector_store.similarity_search(query)
        return found_docs
    

if __name__ == "__main__":
    doc_url = "https://dash.plotly.com/"
    indexer = VectorIndexer()
    indexer.load_index()
    # scraper = DocumentationScraper(doc_url)
    # docs = scraper.load_documents()
    # indexer.index_docs(docs)
    
    results = indexer.search("How to create a bar chart in Dash")
    print(results)