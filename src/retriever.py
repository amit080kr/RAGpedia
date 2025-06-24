from typing import List, Dict, Any
import logging
from src.vector_store import ResearchVectorStore

logger = logging.getLogger(__name__)

class ResearchPaperRetriever:
    """
    A class for retrieving relevant research papers based on semantic queries using FAISS.
    
    This retriever uses FAISS vector similarity search to find papers that are semantically
    similar to the user's query, with optional recency-based ranking.
    Embedding model if needed: OpenAI's text-embedding-ada-002
    """
    
    def __init__(self, vector_store: ResearchVectorStore):
        """
        Initialize the retriever with a FAISS vector store.
        
        Args:
            vector_store (FAISSResearchVectorStore): Vector store containing research paper embeddings
        """
        if not hasattr(vector_store, "query_similar"):
            raise TypeError("vector_store must support query_similar()")
        self.vector_store = vector_store
        logger.info("ResearchPaperRetriever initialized.")     
        
    def retrieve_papers(
        self, 
        query: str, 
        k: int = 5, 
        use_recency: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant research papers for a given query.
        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return value error.
        - Short queries (less than 3 characters) MUST return value error.
        Args:
            query (str): User's research query
            k (int): Number of papers to retrieve
            use_recency (bool): Whether to factor in recency when ranking results
            
        Returns:
            List[Dict[str, Any]]: List of relevant research papers with metadata
            
        Raises:
            ValueError: If query is empty or too short
        
        formatted_results
            {
                'rank': 1,
                'title': "title",
                'authors': 'authors',
                'year': year,
                'venue': 'venue',
                'citations': 'n_citation',
                'abstract': 'abstract'),
                'similarity_score': 0.70,
                'paper_id': 'id'            
            }
        """
        if not query or query.strip() == "":
            logger.error("Query cannot be empty or whitespace-only.")
            raise ValueError("Query cannot be empty or whitespace-only.")

        if len(query.strip()) < 3:
            logger.error("Query too short")
            raise ValueError("Query too short")

        logger.info(f"Retrieving {k} papers for query: '{query}' with recency: {use_recency}")

        raw_results = self.vector_store.query_similar(query, k=k, use_recency=use_recency)

        if not raw_results:
            logger.info("No papers found for the given query.")
            return []

        formatted_results: List[Dict[str, Any]] = []

        if not use_recency:
            raw_results.sort(key=lambda x: x['similarity'], reverse=True)

        for i, item in enumerate(raw_results):
            metadata = item['metadata']

            formatted_results.append({
                'rank': i + 1,
                'title': (
                    metadata.get('title')
                    or (
                        item.get("content").split(":", 1)[1].strip()
                        if isinstance(item.get("content"), str) and ":" in item.get("content")
                        else str(item.get("content")) if item.get("content") else 'N/A'
                    )
                ),
                'authors': metadata.get('authors', 'N/A'),
                'year': int(metadata['year']) if 'year' in metadata and str(metadata['year']).isdigit() else 'N/A',
                'venue': metadata.get('venue', 'N/A'),
                'citations': metadata.get('n_citation', 'N/A'),
                'abstract': metadata.get('abstract', 'N/A'),
                'similarity_score': item['similarity'],
                'paper_id': metadata.get('id', 'N/A')
            })


        logger.info(f"Retrieved and formatted {len(formatted_results)} papers.")
        return formatted_results        

    def retrieve_papers_with_recency(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve research papers with recency factored into ranking.
        
        Args:
            query (str): User's research query
            k (int): Number of papers to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of relevant research papers with metadata in descending order of publication year
        """
        logger.info(f"Retrieving {k} papers for query: '{query}' with recency prioritization.")
        return self.retrieve_papers(query, k=k, use_recency=True)


        
        