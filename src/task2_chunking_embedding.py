"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing

Objective: Convert cleaned text narratives into a format suitable for efficient semantic search.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pickle
import json
from datetime import datetime

# Text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector stores
import faiss
# Alternative: import chromadb (if using ChromaDB)

import warnings
warnings.filterwarnings('ignore')


class ComplaintVectorStore:
    """
    Class to handle chunking, embedding, and vector store creation for complaint data.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_type: str = "faiss"  # or "chromadb"
    ):
        """
        Initialize the vector store builder.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chunk_size: Maximum size of text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            vector_store_type: Type of vector store ("faiss" or "chromadb")
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_type = vector_store_type
        
        # Initialize components
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Storage for chunks and metadata
        self.chunks = []
        self.metadata = []
        self.embeddings = None
        
    def create_stratified_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 12000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create a stratified sample ensuring proportional representation across products.
        
        Args:
            df: Filtered complaint dataframe
            sample_size: Target sample size (10K-15K)
            random_state: Random seed for reproducibility
            
        Returns:
            Stratified sample dataframe
        """
        print(f"\nCreating stratified sample of {sample_size:,} complaints...")
        
        if 'product' not in df.columns:
            raise ValueError("DataFrame must have a 'product' column")
        
        # Get product distribution
        product_counts = df['product'].value_counts()
        print(f"\nOriginal product distribution:")
        for product, count in product_counts.items():
            print(f"  {product}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Calculate proportional sample sizes per product
        product_samples = {}
        total_available = len(df)
        
        for product in product_counts.index:
            proportion = product_counts[product] / total_available
            product_sample_size = int(sample_size * proportion)
            product_samples[product] = min(product_sample_size, product_counts[product])
        
        # Adjust if total is less than sample_size
        total_allocated = sum(product_samples.values())
        if total_allocated < sample_size:
            # Distribute remaining slots to products with available data
            remaining = sample_size - total_allocated
            for product in product_counts.index:
                if remaining <= 0:
                    break
                additional = min(remaining, product_counts[product] - product_samples[product])
                product_samples[product] += additional
                remaining -= additional
        
        # Sample from each product
        sampled_dfs = []
        np.random.seed(random_state)
        
        for product, n_samples in product_samples.items():
            product_df = df[df['product'] == product]
            if len(product_df) >= n_samples:
                sampled = product_df.sample(n=n_samples, random_state=random_state)
            else:
                sampled = product_df  # Use all available
            sampled_dfs.append(sampled)
            print(f"  {product}: {len(sampled):,} samples")
        
        df_sampled = pd.concat(sampled_dfs, ignore_index=True)
        df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        print(f"\nFinal stratified sample: {len(df_sampled):,} complaints")
        print(f"\nSampled product distribution:")
        sampled_dist = df_sampled['product'].value_counts()
        for product, count in sampled_dist.items():
            print(f"  {product}: {count:,} ({count/len(df_sampled)*100:.2f}%)")
        
        return df_sampled
    
    def chunk_texts(self, df: pd.DataFrame, narrative_col: str = 'narrative') -> None:
        """
        Chunk complaint narratives into smaller pieces.
        
        Args:
            df: Dataframe with complaint narratives
            narrative_col: Name of the column containing narratives
        """
        print(f"\nChunking texts (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        if narrative_col not in df.columns:
            raise ValueError(f"Column '{narrative_col}' not found in dataframe")
        
        self.chunks = []
        self.metadata = []
        
        for idx, row in df.iterrows():
            complaint_id = row.get('complaint_id', idx)
            narrative = str(row[narrative_col])
            
            # Skip empty narratives
            if not narrative or narrative.strip() == '' or narrative == 'nan':
                continue
            
            # Split into chunks
            text_chunks = self.text_splitter.split_text(narrative)
            
            # Store chunks with metadata
            for chunk_idx, chunk in enumerate(text_chunks):
                self.chunks.append(chunk)
                
                # Create metadata for this chunk
                chunk_metadata = {
                    'complaint_id': complaint_id,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks),
                    'product_category': row.get('product', 'Unknown'),
                    'product': row.get('product', 'Unknown'),
                    'issue': row.get('issue', 'Unknown'),
                    'sub_issue': row.get('sub_issue', 'Unknown'),
                    'company': row.get('company', 'Unknown'),
                    'state': row.get('state', 'Unknown'),
                    'date_received': str(row.get('date_received', 'Unknown')),
                    'chunk_text': chunk[:100] + '...' if len(chunk) > 100 else chunk  # Preview
                }
                self.metadata.append(chunk_metadata)
        
        print(f"Created {len(self.chunks):,} chunks from {len(df):,} complaints")
        print(f"Average chunks per complaint: {len(self.chunks)/len(df):.2f}")
    
    def generate_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for all text chunks.
        
        Args:
            batch_size: Batch size for embedding generation
            
        Returns:
            Numpy array of embeddings
        """
        print(f"\nGenerating embeddings (batch_size={batch_size})...")
        
        if not self.chunks:
            raise ValueError("No chunks available. Run chunk_texts() first.")
        
        # Generate embeddings in batches
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def create_faiss_index(self, index_type: str = "L2") -> faiss.Index:
        """
        Create a FAISS index from embeddings.
        
        Args:
            index_type: Type of FAISS index ("L2" for L2 distance, "IP" for inner product)
            
        Returns:
            FAISS index
        """
        print(f"\nCreating FAISS index (type={index_type})...")
        
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run generate_embeddings() first.")
        
        # Normalize embeddings for cosine similarity (optional, but recommended)
        # For cosine similarity, we normalize and use inner product
        if index_type == "cosine":
            # Normalize embeddings
            faiss.normalize_L2(self.embeddings)
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
        elif index_type == "L2":
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        index.add(self.embeddings.astype('float32'))
        
        print(f"FAISS index created with {index.ntotal:,} vectors")
        return index
    
    def save_vector_store(
        self,
        output_dir: Path,
        index: faiss.Index = None
    ) -> None:
        """
        Save the vector store and metadata to disk.
        
        Args:
            output_dir: Directory to save the vector store
            index: FAISS index (if using FAISS)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving vector store to {output_dir}...")
        
        if self.vector_store_type == "faiss":
            if index is None:
                raise ValueError("FAISS index required for saving")
            
            # Save FAISS index
            faiss_path = output_dir / "faiss_index.bin"
            faiss.write_index(index, str(faiss_path))
            print(f"Saved FAISS index: {faiss_path}")
            
            # Save embeddings (optional, for backup)
            embeddings_path = output_dir / "embeddings.npy"
            np.save(embeddings_path, self.embeddings)
            print(f"Saved embeddings: {embeddings_path}")
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata: {metadata_path}")
        
        # Save chunks (for reference)
        chunks_path = output_dir / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Saved chunks: {chunks_path}")
        
        # Save configuration
        config = {
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'vector_store_type': self.vector_store_type,
            'num_chunks': len(self.chunks),
            'created_at': datetime.now().isoformat()
        }
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration: {config_path}")
        
        print(f"\nVector store saved successfully!")


def main():
    """
    Main function to run the chunking, embedding, and indexing pipeline.
    """
    # Define paths
    data_dir = Path('../data/processed')
    output_dir = Path('../vector_store')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load filtered dataset
    filtered_data_path = data_dir / 'filtered_complaints.csv'
    
    if not filtered_data_path.exists():
        print(f"Error: Filtered dataset not found at {filtered_data_path}")
        print("Please run Task 1 first to create the filtered dataset.")
        return
    
    print("Loading filtered complaint dataset...")
    df = pd.read_csv(filtered_data_path)
    print(f"Loaded {len(df):,} complaints")
    
    # Initialize vector store builder
    vector_store = ComplaintVectorStore(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        vector_store_type="faiss"
    )
    
    # Step 1: Create stratified sample
    df_sampled = vector_store.create_stratified_sample(
        df=df,
        sample_size=12000,  # Between 10K-15K
        random_state=42
    )
    
    # Step 2: Chunk texts
    vector_store.chunk_texts(df_sampled, narrative_col='narrative')
    
    # Step 3: Generate embeddings
    vector_store.generate_embeddings(batch_size=32)
    
    # Step 4: Create FAISS index
    index = vector_store.create_faiss_index(index_type="cosine")
    
    # Step 5: Save vector store
    vector_store.save_vector_store(output_dir, index=index)
    
    print("\n" + "="*60)
    print("Task 2 completed successfully!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Original complaints: {len(df):,}")
    print(f"  - Sampled complaints: {len(df_sampled):,}")
    print(f"  - Total chunks: {len(vector_store.chunks):,}")
    print(f"  - Embedding dimension: {vector_store.embedding_dim}")
    print(f"  - Vector store saved to: {output_dir}")


if __name__ == "__main__":
    main()
