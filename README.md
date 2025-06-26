# Academic Research Paper Finder

A semantic search engine for research papers using FAISS for efficient similarity search. Enables researchers to find contextually relevant papers beyond keyword matching.

## Features

- **Semantic Search**: Finds conceptually similar papers using HuggingFace embeddings
- **Recency Ranking**: Optionally prioritize newer research (70% similarity + 30% recency)
- **Efficient Retrieval**: FAISS-optimized for fast search over large paper collections
- **CSV Flexibility**: Accepts various column name formats (e.g., "Title" or "Publication Year")

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amit080kr/RAGpedia.git
   cd RAGpedia

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Data Preparation
Place your research papers in data/dataset.csv with these columns (case-insensitive):
```bash

| Column          | Example                | Required |
|-----------------|----------------------- |----------|
| Title           | "Quantum Cryptography" |    Yes   |
| Abstract        | "A new method..."      |    Yes   |
| Authors         | "Alice, Bob"           |    No    |
| Year            | 2023                   |    No    |
| Venue           | "Nature"               |    No    |
| Citation Count  | 42                     |    No    |

> 📝 Note: Column names are case-insensitive (e.g., "TITLE" or "title" both work) and can contain any columns.

```
5. Usage
  ```bash
   streamlit run app.py
  ```

#### Interface Options:

- Enter research topic/question
- Specify the number of papers to retrieve
- Toggle recency prioritization

![flow_chat](https://github.com/user-attachments/assets/b2ee3d14-6d1c-4cc4-93bc-01f51bd09050)

## File Structure

```bash
   research-paper-finder/
   ├── app.py               # Streamlit interface
   ├── src/
   │   ├── document_loader.py # CSV processing
   │   ├── retriever.py     # Search logic
   │   └── vector_store.py  # FAISS operations
   ├── data/                # Your papers go here
   ├── faiss_store/         # Auto-generated index
   └── requirements.txt
```
Tip:
For large paper collections, consider using GPU-accelerated FAISS:
   ```bash
  pip install faiss-gpu
