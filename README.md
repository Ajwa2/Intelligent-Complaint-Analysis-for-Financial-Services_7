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

### Task 3: RAG Pipeline and Evaluation

1. **Ensure you have the pre-built embeddings**:

   - Pre-built embeddings file: `data/data/complaint_embeddings-002.parquet`
   - Or your own vector store from Task 2 in `vector_store/`

2. **Run the evaluation script**:

   ```bash
   python src/task3_evaluation.py
   ```

3. **Outputs**:
   - Evaluation results: `notebooks/evaluation_results.csv` and `.md`
   - Fill in `notebooks/TASK3_EVALUATION_REPORT.md` with quality scores and analysis

### Task 4: Interactive Chat Interface

1. **Launch the Gradio interface**:

   ```bash
   python app.py
   ```

2. **The interface will**:

   - Open in your browser at `http://localhost:7860`
   - Allow you to ask questions about customer complaints
   - Display answers with source citations
   - Show retrieved complaint excerpts for transparency

3. **Features**:
   - Natural language question input
   - Real-time answer generation
   - Source display for verification
   - Example questions provided
   - Clear chat functionality

## Project Tasks

- ✅ **Task 1**: EDA and Data Preprocessing
- ✅ **Task 2**: Text Chunking, Embedding, and Vector Store Indexing
- ✅ **Task 3**: RAG Pipeline Implementation and Evaluation
- ✅ **Task 4**: Interactive Chat Interface (Gradio)
