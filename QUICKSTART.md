# Quick Start Guide

## Prerequisites

1. Python 3.9 or higher
2. Virtual environment (recommended)

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### Option 1: Use Pre-built Embeddings (Recommended)

If you have the pre-built embeddings file (`data/data/complaint_embeddings-002.parquet`):

```bash
# Simply run the app
python app.py
```

The app will automatically:
- Load the pre-built embeddings
- Create a FAISS index
- Launch the Gradio interface

### Option 2: Build Your Own Vector Store

If you want to build the vector store from scratch:

1. **Run Task 1** (EDA and Preprocessing):
   ```bash
   jupyter notebook notebooks/task1_eda_preprocessing.ipynb
   ```
   - This creates `data/processed/filtered_complaints.csv`

2. **Run Task 2** (Chunking and Embedding):
   ```bash
   python src/task2_chunking_embedding.py
   ```
   - This creates the vector store in `vector_store/`

3. **Run the App**:
   ```bash
   python app.py
   ```

## Using the Chat Interface

1. Open your browser to `http://localhost:7860`
2. Type a question about customer complaints
3. Click "Ask Question" or press Enter
4. View the answer and source citations

### Example Questions

- "Why are customers unhappy with Credit Cards?"
- "What are the main issues with Personal Loans?"
- "What problems do customers face with Money Transfers?"
- "Which product has the most billing disputes?"

## Troubleshooting

### Issue: "Vector store not found"

**Solution**: Ensure one of the following exists:
- `data/data/complaint_embeddings-002.parquet` (pre-built embeddings)
- `vector_store/faiss_index.bin` and `vector_store/metadata.json` (your own vector store)

### Issue: "Module not found"

**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: App is slow to start

**Solution**: This is normal. The first run needs to:
- Load the embedding model (~80MB)
- Load/create the vector index (can take 1-2 minutes for large datasets)

### Issue: LLM responses are template-based

**Solution**: Install transformers library for local LLM:
```bash
pip install transformers accelerate
```

For better results, consider using cloud-based LLM APIs (OpenAI, Anthropic, etc.) by modifying `src/rag_pipeline.py`.

## Next Steps

1. **Evaluate the system**: Run `python src/task3_evaluation.py`
2. **Review results**: Check `notebooks/evaluation_results.md`
3. **Customize prompts**: Edit `src/rag_pipeline.py` → `create_prompt()` method
4. **Improve retrieval**: Adjust `top_k` parameter in RAGPipeline initialization

## Support

For issues or questions, refer to:
- `README.md` for detailed documentation
- `notebooks/TASK3_EVALUATION_REPORT.md` for evaluation guidelines
- Project structure documentation in `README.md`
