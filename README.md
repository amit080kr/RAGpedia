# 🔍 Academic Research Paper Finder (FAISS)

A semantic search engine for research papers using FAISS for efficient similarity search. Enables researchers to find contextually relevant papers beyond keyword matching.

![Demo Screenshot](demo.gif) *(Replace with actual screenshot)*

## ✨ Features

- **Semantic Search**: Finds conceptually similar papers using OpenAI embeddings
- **Recency Ranking**: Optionally prioritize newer research (70% similarity + 30% recency)
- **Efficient Retrieval**: FAISS-optimized for fast search over large paper collections
- **CSV Flexibility**: Accepts various column name formats (e.g., "Title" or "Publication Year")

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/research-paper-finder.git
   cd research-paper-finder

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Data Preparation
Place your research papers in data/dataset.csv with these columns (case-insensitive):

4. Usage
  ```bash
  streamlit run app.py

### Interface Options:

**Enter research topic/question**

**Specify number of papers to retrieve (default: 5)**

**Toggle recency prioritization**

![flow_chat](https://github.com/user-attachments/assets/b2ee3d14-6d1c-4cc4-93bc-01f51bd09050)
![flow_chat](https://github.com/user-attachments/assets/b2ee3d14-6d1c-4cc4-93bc-01f51bd09050)
