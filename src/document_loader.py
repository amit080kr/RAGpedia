from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import logging
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from uuid import uuid4

logger = logging.getLogger(__name__)

class ResearchPaperLoader:
    """
    A class to load and process research papers from CSV files.
    
    This loader processes academic research papers and converts them into
    a standardized document format for the vector store and retrieval system.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the document loader with the path to data files.
        
        Args:
            data_path (str): Path to the CSV file containing research papers

        Raise FileNotFoundError if data_path does not exists.

        """
        self.data_path =Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"file not found at {self.data_path}")
        logger.info(f"File Loaded at {self.data_path}")
        
        
        

    def create_documents(self) -> List[Document]:
        """
        Load research papers from CSV and convert to LangChain Document objects.

        **source_column: title**\n
        **metadata_columns: "abstract", "authors", "n_citation", "references", "venue", "year", "id"**
        Returns:
            List[Document]: List of Document objects representing research papers
        """
        logger.info(f"Loading documents from {self.data_path}")
        try:
            loader = CSVLoader(
                file_path=str(self.data_path),
                csv_args={
                    "delimiter": ",",
                },
                source_column="title"
            )

            documents = loader.load()
            df = pd.read_csv(self.data_path)

            processed_documents: List[Document] = []
            metadata_columns_to_extract = ["title", "abstract", "authors", "n_citation", "references", "venue", "year", "id"]

            for i, doc in enumerate(documents):
                if i >= len(df):
                    logger.warning(f"Row index {i} out of bounds for DataFrame. Skipping document.")
                    continue

                original_row = df.iloc[i]
                new_metadata = {col: original_row[col] for col in metadata_columns_to_extract if col in original_row}

                for key in new_metadata:
                    if pd.notna(new_metadata[key]):
                        new_metadata[key] = str(new_metadata[key])
                    else:
                        new_metadata[key] = "N/A"

                processed_documents.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=new_metadata
                    )
                )

            logger.info(f"Successfully loaded and processed {len(processed_documents)} documents.")
            return processed_documents

        except Exception as e:
            logger.error(f"Error loading or processing documents: {e}", exc_info=True)
            return None