from typing import List, Dict, Any, Optional
import os
import faiss
import numpy as np
import pickle
from pathlib import Path
import logging
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import pandas as pd
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

class ResearchVectorStore:
    """
    A class to manage the vector store for research papers using FAISS.
    
    This class handles the creation, storage, and retrieval of vector embeddings
    for academic research papers.
    """
    
    def __init__(self, store_path: str):
        """
        Initialize the vector store with FAISS and embeddings model.
        
        Args:
            store_path (str): Path where the vector store will be saved

        Note: 
        1. Initialize store_path as self.store_path using Path
        2. Create a directory for the store if it doesn't exist
        3. initialize embedding as self.embeddings, index as self.index, documents as self.documents
        4. initialize metadata as self.metadata, embedding size as self.embedding_size
        5. Embedding Model to use: OpenAI's text-embedding-ada-002
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.embedding_size = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        self.index: Optional[faiss.IndexFlatL2] = None
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        logger.info(f"Research Vector store at {self.store_path}")

    @property
    def embedding_dim(self) -> int:
        if self.embedding_size is None:
            dummy_embedding = self.embeddings.embed_query("test")
            self.embedding_size = len(dummy_embedding)
        return self.embedding_size
        
    def _get_embedding_size(self)   -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            int: Dimensionality of the embeddings
        """
        if self.embedding_size is None:
            dummy_embedding = self.embeddings.embed_query("test")
            self.embedding_size = len(dummy_embedding)
        return self.embedding_size
        

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create vector store from research paper documents.
        
        Args:
            documents (List[Document]): List of research paper documents

        Note:
        1. store documents and metadata to self.documents and self.metadata
        2. embedd the documents
        3. initialize FAISS index as self.index and add embedings to index: you can consider performing L2 normalization to embeddings and cast to float32
        4. save the vector store using save function. (self.save to be implemented below.)
        """
        logger.info(f"Creating vector store for {len(documents)} documents.")
        self.documents = [doc.page_content for doc in documents]
        self.metadata = [doc.metadata for doc in documents]

        self.embedding_size = self.embedding_dim

        logger.info("embedding generating..")
        document_contents = [doc.page_content for doc in documents]
        embeddings_list = self.embeddings.embed_documents(document_contents)
        embeddings_array = np.array(embeddings_list).astype('float32')

        faiss.normalize_L2(embeddings_array)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_size)
        self.index.add(embeddings_array)

        logger.info("Embeddings generated and added to FAISS index.")
        self.save()
        logger.info("Vectr saved successfully.")
        
        
    def save(self) -> None:
        """Save the vector store to disk
        save FAISS index on index_path: faiss_index.bin
        save metadata on metadata_path: metadata.pkl
        save documents on document_path: documents.pkl


        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized. Cannot save an empty vector store.")

        index_path = self.store_path / "faiss_index.bin"
        metadata_path = self.store_path / "metadata.pkl"
        documents_path = self.store_path / "documents.pkl"

        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, str(index_path))

        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saving documents to {documents_path}")
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info("Vector store saved successfully.")
        
        
    @classmethod
    def load(cls, store_path: str) -> 'ResearchVectorStore':
        """
        Load vector store from disk.
        
        Args:
            store_path (str): Path to the stored vector data
            
        Returns:
            FAISSResearchVectorStore: Loaded vector store

        Note:
        1. check if all files exists. 
        2. load faiss index as instance.index and load metadata and documents
        3. Get embedding dimension from index
        """
        store_path_obj = Path(store_path)
        index_path = store_path_obj / "faiss_index.bin"
        metadata_path = store_path_obj / "metadata.pkl"
        documents_path = store_path_obj / "documents.pkl"

        if not all(p.exists() for p in [index_path, metadata_path, documents_path]):
            raise FileNotFoundError(f"files not found in {store_path}. "
                                    "make sure faiss_index.bin, metadata.pkl, and documents.pkl exist.")

        logger.info(f"FAISS index loading from {index_path}")
        index = faiss.read_index(str(index_path))

        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        logger.info(f"documents loading from {documents_path}")
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)

        instance = cls(store_path)
        instance.index = index
        instance.documents = documents
        instance.metadata = metadata
        instance.embedding_size = index.d 

        logger.info(f"Vctor loaded succesfully from {store_path}.")
        return instance

        
        
    def query_similar(self, query: str, k: int = 5, use_recency: bool = False) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar research papers.
        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return an empty list
        - When the query is null or whitespace-only, log a warning but DO NOT raise an exception
        - Non-existent support types MUST return an empty list with an appropriate warning
    
        Args:
            query (str): Query text to find similar documents
            k (int): Number of similar documents to return
            use_recency (bool): Whether to factor in recency of papers when ranking
                When True, results are ranked using a combined score that weighs both 
                semantic similarity (70%) and publication recency (30%).
                
        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata

            format for dictionary:
            {
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarity)
            }
            
        Notes:
            When use_recency=True, a combined score is calculated as follows:
            1. Semantic similarity score: Cosine similarity between query and paper (0-1)
            2. Recency score: Normalized score based on publication year
            recency_score = (paper_year - (current_year - 100)) / 100
            This creates a 0-1 score where recent papers score higher
            3. Combined score: 0.7 * similarity_score + 0.3 * recency_score
            4. Results are sorted by this combined score in descending order
            
            If paper year is missing, only the similarity score is used.

        """
        if not query or query.strip() == "":
            logger.warning("Empty or whitespace-only query received. Returning empty list.")
            return []

        if self.index is None or not self.documents or not self.metadata:
            logger.warning("Vector store is not initialized or contains no documents. Cannot perform query.")
            return []

        try:
            query_embedding = np.array(self.embeddings.embed_query(query)).astype('float32')
            faiss.normalize_L2(query_embedding.reshape(1, -1)) # Reshape for FAISS search

            # D is distances, I is indices
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1: # FAISS returns -1 for unpopulated indices
                    continue

                similarity = 1 - distances[0][i] # Convert L2 distance to similarity (0-1)

                paper_metadata = self.metadata[idx]
                paper_content = self.documents[idx]

                result_item = {
                    'content': paper_content,
                    'metadata': paper_metadata,
                    'similarity': float(similarity)
                }
                results.append(result_item)

            if use_recency:
                # Apply recency scoring and re-sort
                current_year = 2025 # Assuming current year as per problem context

                for item in results:
                    year = item['metadata'].get('year')

                    if pd.isna(year): # Check for NaN or None, assuming pandas might load it as NaN
                        # If year is missing, recency score is 0, only similarity matters
                        recency_score = 0.0
                        combined_score = item['similarity'] 
                        logger.warning(f"Year missing for document {item['metadata'].get('id')}. Recency score not applied.")
                    else:
                        try:
                            year = int(year) # Ensure year is an integer
                            # Normalize year to a 0-1 range based on a 100-year span
                            recency_score = (year - (current_year - 100)) / 100
                            recency_score = max(0.0, min(1.0, recency_score)) # Clamp between 0 and 1

                            combined_score = 0.7 * item['similarity'] + 0.3 * recency_score
                        except (ValueError, TypeError):
                            # Handle cases where year might be malformed or non-numeric
                            recency_score = 0.0
                            combined_score = item['similarity']
                            logger.warning(f"Invalid year '{year}' for document {item['metadata'].get('id')}. Recency score not applied.")

                    item['combined_score'] = combined_score

                # Sort by combined score in descending order
                results.sort(key=lambda x: x.get('combined_score', x['similarity']), reverse=True)

            return results

        except Exception as e:
            logger.error(f"Error during query_similar: {e}", exc_info=True)
            return []

        