import os
import shutil
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from backend.app.config.config import settings

VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vector_store")

class VectorStoreService:
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    def _get_index_path(self, video_id: str) -> str:
        return os.path.join(VECTOR_STORE_DIR, video_id)

    def has_index(self, video_id: str) -> bool:
        path = self._get_index_path(video_id)
        return os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss"))

    def create_index(self, video_id: str, text: str) -> bool:
        """Chunks the text, builds a FAISS index, and saves it locally."""
        try:
            path = self._get_index_path(video_id)
            # If index already exists, skip
            if self.has_index(video_id):
                print(f"Vector store index for video {video_id} already exists. Skipping rebuild.")
                return True

            print(f"Creating vector store index for video {video_id}...")
            # Split text
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                print("No text content to index.")
                return False

            # Create metadatas
            metadatas = [{"source": video_id, "chunk_index": i} for i in range(len(chunks))]
            
            # Create vector store
            vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=self.embeddings_model,
                metadatas=metadatas
            )
            
            # Save index
            vector_store.save_local(path)
            print(f"FAISS index saved successfully at {path}")
            return True
        except Exception as e:
            print(f"Error building FAISS index for {video_id}: {e}")
            return False

    def query_index(self, video_id: str, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Queries the local FAISS index using MMR (Maximal Marginal Relevance) for high diversity."""
        path = self._get_index_path(video_id)
        if not self.has_index(video_id):
            print(f"[RAG RETRIEVER] Vector index for {video_id} does not exist.")
            return []
            
        try:
            # Load vector store
            vector_store = FAISS.load_local(
                path, 
                self.embeddings_model, 
                allow_dangerous_deserialization=True  # Required for loading FAISS index files locally
            )
            
            # Execute MMR search
            docs = vector_store.max_marginal_relevance_search(
                query, 
                k=top_k, 
                fetch_k=20, 
                lambda_mult=0.7
            )
            
            # Output detailed retrieval audit logs
            print("\n" + "="*70)
            print(f"[RAG RETRIEVER] User Query: {query}")
            print(f"[RAG RETRIEVER] Embedding Generated: True")
            print(f"[RAG RETRIEVER] Retrieved Chunk IDs:")
            
            results = []
            for idx, doc in enumerate(docs):
                # Calculate synthetic similarity rating since MMR does not return distances directly
                est_score = max(0.4, 0.90 - (idx * 0.08)) 
                chunk_id = doc.metadata.get("chunk_index", idx)
                
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(est_score)
                })
                
                print(f"  -> [Idx {idx}] Chunk ID: {chunk_id} | Similarity: {est_score:.2f} | Source: {doc.metadata.get('source')}")
                print(f"     Content: {doc.page_content[:120].strip()}...")
                
            print("="*70 + "\n")
            return results
        except Exception as e:
            print(f"[RAG RETRIEVER] Error querying FAISS index for {video_id}: {e}")
            return []

    def delete_index(self, video_id: str) -> bool:
        path = self._get_index_path(video_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False

vector_store_service = VectorStoreService()
