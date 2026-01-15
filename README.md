# RAG Complaint Chatbot

Intelligent Complaint Analysis for Financial Services using Retrieval-Augmented Generation (RAG).

## Project Structure

```
rag-complaint-chatbot/
├── .vscode/              # VS Code settings
├── .github/              # GitHub workflows
├── data/                 # Data directories
│   ├── raw/             # Raw data files
│   └── processed/       # Processed data files
├── vector_store/         # Persisted FAISS/ChromaDB index
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
├── tests/                # Unit tests
├── app.py                # Gradio/Streamlit interface
└── requirements.txt      # Python dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Task 1: Exploratory Data Analysis and Preprocessing

1. **Place your CFPB dataset** in the `data/raw/` directory (or `data/` directory)
   - The script will automatically detect CSV files

2. **Run the EDA notebook**:
   ```bash
   jupyter notebook notebooks/task1_eda_preprocessing.ipynb
   ```
   Or execute as a script (if converted):
   ```bash
   python notebooks/task1_eda_preprocessing.py
   ```

3. **Outputs**:
   - Cleaned and filtered dataset: `data/processed/filtered_complaints.csv`
   - Visualization plots in `data/processed/`
   - Fill in `notebooks/EDA_SUMMARY.md` with your findings

### Task 2: Text Chunking, Embedding, and Vector Store

1. **Ensure Task 1 is completed** and `data/processed/filtered_complaints.csv` exists

2. **Run the chunking and embedding script**:
   ```bash
   python src/task2_chunking_embedding.py
   ```

3. **Outputs**:
   - Vector store saved to `vector_store/` directory:
     - `faiss_index.bin`: FAISS vector index
     - `embeddings.npy`: Embedding vectors
     - `metadata.json`: Chunk metadata
     - `chunks.pkl`: Text chunks
     - `config.json`: Configuration
   - Fill in `notebooks/TASK2_REPORT.md` with your methodology and findings

### Task 3-4: RAG Pipeline and UI (Coming Soon)

```bash
python app.py
```

## Project Tasks

- ✅ **Task 1**: EDA and Data Preprocessing
- ✅ **Task 2**: Text Chunking, Embedding, and Vector Store Indexing
- ⏳ **Task 3**: RAG Pipeline Implementation
- ⏳ **Task 4**: User Interface Development

## License

TODO: Add license information
