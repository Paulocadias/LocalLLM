"""
Vector Database Service using ChromaDB

Provides RAG (Retrieval Augmented Generation) capabilities:
- Document ingestion and embedding
- Semantic search and retrieval
- Knowledge base management
"""

import logging
import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vector DB Service (RAG)", version="1.0.0")

# Initialize ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/app/data/chromadb")
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Initialize embedding model (using sentence-transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")


class Document(BaseModel):
    """Document to be stored in vector database"""
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    collection_name: str = "knowledge_base"


class DocumentBatch(BaseModel):
    """Batch of documents to be stored"""
    documents: List[Document]


class SearchQuery(BaseModel):
    """Search query for retrieval"""
    query: str
    collection_name: str = "knowledge_base"
    top_k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Search result with document and similarity score"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    document_id: str


class SearchResponse(BaseModel):
    """Response containing search results"""
    query: str
    results: List[SearchResult]
    search_time: float
    total_results: int


class CollectionInfo(BaseModel):
    """Information about a collection"""
    name: str
    count: int
    metadata: Optional[Dict[str, Any]] = {}


class VectorDBService:
    """Service for managing vector database operations"""

    def __init__(self):
        self.client = chroma_client
        logger.info("VectorDBService initialized")

    def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
            logger.info(f"Collection '{collection_name}' ready (count: {collection.count()})")
            return collection
        except Exception as e:
            logger.error(f"Error getting/creating collection: {e}")
            raise HTTPException(status_code=500, detail=f"Collection error: {str(e)}")

    def add_document(self, document: Document) -> Dict[str, Any]:
        """Add a single document to the vector database"""
        try:
            collection = self.get_or_create_collection(document.collection_name)

            # Generate embedding
            embedding = embedding_model.encode(document.content).tolist()

            # Generate document ID
            doc_id = f"doc_{int(time.time() * 1000)}"

            # Add metadata
            metadata = document.metadata or {}
            metadata["indexed_at"] = datetime.now().isoformat()
            metadata["content_length"] = len(document.content)

            # Store in ChromaDB
            collection.add(
                embeddings=[embedding],
                documents=[document.content],
                metadatas=[metadata],
                ids=[doc_id]
            )

            logger.info(f"Added document {doc_id} to collection '{document.collection_name}'")

            return {
                "document_id": doc_id,
                "collection": document.collection_name,
                "indexed_at": metadata["indexed_at"],
                "content_length": metadata["content_length"]
            }

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

    def add_documents_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """Add multiple documents in batch"""
        try:
            # Group documents by collection
            collections = {}
            for doc in documents:
                if doc.collection_name not in collections:
                    collections[doc.collection_name] = []
                collections[doc.collection_name].append(doc)

            results = []
            for collection_name, docs in collections.items():
                collection = self.get_or_create_collection(collection_name)

                # Generate embeddings for all documents
                contents = [doc.content for doc in docs]
                embeddings = embedding_model.encode(contents).tolist()

                # Generate IDs and metadata
                doc_ids = [f"doc_{int(time.time() * 1000)}_{i}" for i in range(len(docs))]
                metadatas = []
                for doc in docs:
                    metadata = doc.metadata or {}
                    metadata["indexed_at"] = datetime.now().isoformat()
                    metadata["content_length"] = len(doc.content)
                    metadatas.append(metadata)

                # Store in ChromaDB
                collection.add(
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                    ids=doc_ids
                )

                results.extend([
                    {"document_id": doc_id, "collection": collection_name}
                    for doc_id in doc_ids
                ])

            logger.info(f"Added {len(results)} documents in batch")

            return {
                "documents_added": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error adding batch: {e}")
            raise HTTPException(status_code=500, detail=f"Batch add failed: {str(e)}")

    def search(self, query: SearchQuery) -> SearchResponse:
        """Search for similar documents"""
        start_time = time.time()

        try:
            collection = self.get_or_create_collection(query.collection_name)

            # Generate query embedding
            query_embedding = embedding_model.encode(query.query).tolist()

            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=query.top_k,
                where=query.filter_metadata
            )

            # Format results
            search_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    search_results.append(SearchResult(
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                        similarity_score=1.0 - results['distances'][0][i] if results['distances'] else 0.0,
                        document_id=results['ids'][0][i]
                    ))

            search_time = time.time() - start_time

            logger.info(
                f"Search completed: query='{query.query}', "
                f"results={len(search_results)}, time={search_time:.3f}s"
            )

            return SearchResponse(
                query=query.query,
                results=search_results,
                search_time=search_time,
                total_results=len(search_results)
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def list_collections(self) -> List[CollectionInfo]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            return [
                CollectionInfo(
                    name=col.name,
                    count=col.count(),
                    metadata=col.metadata
                )
                for col in collections
            ]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")

    def delete_collection(self, collection_name: str) -> Dict[str, str]:
        """Delete a collection"""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return {"status": "deleted", "collection": collection_name}
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# Initialize service
vector_service = VectorDBService()


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        collections = vector_service.list_collections()
        return {
            "status": "healthy",
            "service": "vector-db",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "collections_count": len(collections)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/documents/add")
async def add_document(document: Document):
    """Add a single document to the vector database"""
    result = vector_service.add_document(document)
    return {"status": "success", **result}


@app.post("/documents/add-batch")
async def add_documents_batch(batch: DocumentBatch):
    """Add multiple documents in batch"""
    result = vector_service.add_documents_batch(batch.documents)
    return {"status": "success", **result}


@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Search for similar documents using semantic similarity"""
    return vector_service.search(query)


@app.get("/collections")
async def list_collections():
    """List all collections"""
    collections = vector_service.list_collections()
    return {
        "collections": collections,
        "total": len(collections)
    }


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    return vector_service.delete_collection(collection_name)


@app.get("/collections/{collection_name}/stats")
async def get_collection_stats(collection_name: str):
    """Get statistics for a collection"""
    try:
        collection = vector_service.get_or_create_collection(collection_name)
        return {
            "name": collection_name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
