from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import logging
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)

class ResearchPaperLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found at {self.data_path}")
        logger.info(f"File loaded at {self.data_path}")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to handle case and common variations."""
        column_mapping = {
            'title': ['title', 'paper title', 'name'],
            'authors': ['authors', 'author', 'creators'],
            'year': ['year', 'publication year', 'date', 'pub year'],
            'abstract': ['abstract', 'abstract note', 'summary', 'description'],
            'venue': ['venue', 'journal', 'publication', 'conference'],
            'n_citation': ['n_citation', 'citations', 'citation count']
        }
        
        # Create case-insensitive mapping
        normalized_columns = {}
        for standard_name, alternatives in column_mapping.items():
            for col in df.columns:
                if col.lower() in [alt.lower() for alt in alternatives]:
                    normalized_columns[col] = standard_name
                    break
        
        return df.rename(columns=normalized_columns)

    def create_documents(self) -> List[Document]:
        try:
            # Load and normalize columns
            df = pd.read_csv(self.data_path)
            df = self._normalize_columns(df)
            
            # Validate required columns
            required_columns = {'title', 'abstract'}
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns after normalization: {missing}")

            # Create documents with standardized column names
            documents = []
            metadata_columns = ['title', 'authors', 'year', 'abstract', 'venue', 'n_citation']
            
            for _, row in df.iterrows():
                metadata = {
                    'title': str(row.get('title', 'N/A')),
                    'authors': str(row.get('authors', 'N/A')),
                    'year': str(row.get('year', 'N/A')),
                    'abstract': str(row.get('abstract', 'N/A')),
                    'venue': str(row.get('venue', 'N/A')),
                    'n_citation': str(row.get('n_citation', 'N/A')),
                    'id': str(row.get('id', 'N/A'))  # Optional
                }
                
                documents.append(
                    Document(
                        page_content=f"{metadata['title']}\n{metadata['abstract']}",
                        metadata=metadata
                    )
                )
            
            logger.info(f"Processed {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error creating documents: {e}", exc_info=True)
            return None