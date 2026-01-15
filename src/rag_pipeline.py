"""
Task 3: RAG Pipeline Implementation

This module implements the Retrieval-Augmented Generation pipeline for the complaint chatbot.
"""

import pandas as pd
import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for complaint analysis.
    """
    
    def __init__(
        self,
        vector_store_dir: Path = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store_dir: Directory containing the vector store files
            embedding_model_name: Name of the embedding model
            top_k: Number of top chunks to retrieve
        """
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Load vector store
        if vector_store_dir is None:
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            vector_store_dir = project_root / 'vector_store'
        else:
            vector_store_dir = Path(vector_store_dir)
        
        self.vector_store_dir = vector_store_dir
        self._load_vector_store()
        
        # LLM will be initialized when needed (lazy loading)
        self.llm = None
        self.llm_initialized = False
    
    def _load_vector_store(self):
        """Load the pre-built vector store and metadata."""
        print(f"Loading vector store from {self.vector_store_dir}...")
        
        # Try to load from pre-built embeddings parquet file first
        project_root = Path(__file__).parent.parent
        embeddings_file = project_root / 'data' / 'data' / 'complaint_embeddings-002.parquet'
        if embeddings_file.exists():
            print("Loading from pre-built embeddings parquet file...")
            self._load_from_parquet(embeddings_file)
        else:
            # Try loading from vector_store directory
            faiss_index_path = self.vector_store_dir / 'faiss_index.bin'
            metadata_path = self.vector_store_dir / 'metadata.json'
            chunks_path = self.vector_store_dir / 'chunks.pkl'
            
            if faiss_index_path.exists() and metadata_path.exists():
                print("Loading from vector_store directory...")
                self._load_from_files(faiss_index_path, metadata_path, chunks_path)
            else:
                raise FileNotFoundError(
                    f"Vector store not found. Please ensure either:\n"
                    f"1. Pre-built embeddings file exists at: {embeddings_file}\n"
                    f"2. Or vector store files exist in: {self.vector_store_dir}"
                )
    
    def _load_from_parquet(self, embeddings_file: Path):
        """Load vector store from pre-built parquet file."""
        print(f"Reading parquet file: {embeddings_file}")
        df_embeddings = pd.read_parquet(embeddings_file)
        
        print(f"Loaded {len(df_embeddings):,} chunks from parquet")
        print(f"Columns in parquet file: {df_embeddings.columns.tolist()}")
        
        # Extract embeddings - handle different possible formats
        embeddings_list = []
        
        # Try different possible column names for embeddings
        embedding_cols = [col for col in df_embeddings.columns 
                         if 'embedding' in col.lower() or 'vector' in col.lower() or 'emb' in col.lower()]
        
        if not embedding_cols:
            # If no embedding column found, check if we need to generate them
            # or if they're stored in a different format
            print("Warning: No embedding column found. Checking data structure...")
            print("First few rows sample:")
            print(df_embeddings.head())
            
            # Try to find any array-like columns
            for col in df_embeddings.columns:
                sample_val = df_embeddings[col].iloc[0]
                if isinstance(sample_val, (list, np.ndarray)) or (hasattr(sample_val, '__len__') and len(sample_val) > 10):
                    embedding_cols = [col]
                    print(f"Found potential embedding column: {col}")
                    break
        
        if embedding_cols:
            col = embedding_cols[0]
            print(f"Extracting embeddings from column: {col}")
            
            for idx, emb in enumerate(df_embeddings[col]):
                try:
                    if isinstance(emb, np.ndarray):
                        embeddings_list.append(emb.astype('float32'))
                    elif isinstance(emb, list):
                        embeddings_list.append(np.array(emb, dtype='float32'))
                    elif isinstance(emb, str):
                        # Try to parse JSON string
                        try:
                            parsed = json.loads(emb)
                            embeddings_list.append(np.array(parsed, dtype='float32'))
                        except:
                            # If it's a space-separated string
                            embeddings_list.append(np.array(emb.split(), dtype='float32'))
                    else:
                        # Try to convert to array
                        embeddings_list.append(np.array(emb, dtype='float32'))
                except Exception as e:
                    print(f"Warning: Could not parse embedding at index {idx}: {e}")
                    # Use zero vector as fallback
                    embeddings_list.append(np.zeros(self.embedding_dim, dtype='float32'))
            
            self.embeddings = np.vstack(embeddings_list).astype('float32')
            print(f"Extracted embeddings shape: {self.embeddings.shape}")
        else:
            raise ValueError(
                "Could not find embedding column in parquet file. "
                f"Available columns: {df_embeddings.columns.tolist()}"
            )
        
        # Extract chunks and metadata
        chunk_text_cols = [col for col in df_embeddings.columns 
                          if 'text' in col.lower() or 'chunk' in col.lower() or 'narrative' in col.lower()]
        
        if chunk_text_cols:
            chunk_col = chunk_text_cols[0]
            self.chunks = df_embeddings[chunk_col].astype(str).tolist()
            print(f"Extracted chunks from column: {chunk_col}")
        else:
            # If no text column, try to construct from metadata
            print("Warning: No chunk text column found. Attempting to construct from available data...")
            # Fallback: use first text-like column or create placeholder
            text_like_cols = [col for col in df_embeddings.columns if df_embeddings[col].dtype == 'object']
            if text_like_cols:
                self.chunks = df_embeddings[text_like_cols[0]].astype(str).tolist()
                print(f"Using column '{text_like_cols[0]}' as chunk text")
            else:
                raise ValueError("Could not find chunk text column in parquet file")
        
        # Extract metadata
        metadata_cols = ['complaint_id', 'product_category', 'product', 'issue', 'sub_issue', 
                        'company', 'state', 'date_received', 'chunk_index', 'total_chunks']
        self.metadata = []
        
        for idx, row in df_embeddings.iterrows():
            meta = {}
            for col in metadata_cols:
                if col in df_embeddings.columns:
                    val = row[col]
                    meta[col] = str(val) if pd.notna(val) else 'Unknown'
                else:
                    meta[col] = 'Unknown'
            meta['chunk_text'] = self.chunks[idx] if idx < len(self.chunks) else ''
            self.metadata.append(meta)
        
        print(f"Extracted metadata for {len(self.metadata):,} chunks")
        
        # Create FAISS index
        print("Creating FAISS index...")
        # Normalize embeddings for cosine similarity
        if self.embeddings.shape[1] != self.embedding_dim:
            print(f"Warning: Embedding dimension mismatch. Expected {self.embedding_dim}, got {self.embeddings.shape[1]}")
            # Try to adjust
            if self.embeddings.shape[1] > self.embedding_dim:
                self.embeddings = self.embeddings[:, :self.embedding_dim]
            else:
                # Pad with zeros
                padding = np.zeros((self.embeddings.shape[0], self.embedding_dim - self.embeddings.shape[1]), dtype='float32')
                self.embeddings = np.hstack([self.embeddings, padding])
        
        faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings)
        print(f"✓ FAISS index created with {self.index.ntotal:,} vectors")
    
    def _load_from_files(self, faiss_index_path: Path, metadata_path: Path, chunks_path: Path):
        """Load vector store from saved files."""
        import pickle
        
        # Load FAISS index
        self.index = faiss.read_index(str(faiss_index_path))
        print(f"Loaded FAISS index with {self.index.ntotal:,} vectors")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"Loaded metadata for {len(self.metadata):,} chunks")
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {len(self.chunks):,} text chunks")
        
        # Load embeddings if available
        embeddings_path = self.vector_store_dir / 'embeddings.npy'
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        else:
            self.embeddings = None
    
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a given question.
        
        Args:
            question: User's question string
            
        Returns:
            List of dictionaries containing retrieved chunks and metadata
        """
        # Embed the question
        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(question_embedding, self.top_k)
        
        # Retrieve chunks and metadata
        retrieved_chunks = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks) and idx < len(self.metadata):
                chunk_info = {
                    'rank': i + 1,
                    'similarity_score': float(dist),
                    'chunk_text': self.chunks[idx],
                    'metadata': self.metadata[idx].copy()
                }
                retrieved_chunks.append(chunk_info)
        
        return retrieved_chunks
    
    def create_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the LLM using retrieved context.
        
        Args:
            question: User's question
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted prompt string
        """
        # Format context from retrieved chunks
        context_parts = []
        for i, chunk_info in enumerate(retrieved_chunks, 1):
            chunk_text = chunk_info['chunk_text']
            metadata = chunk_info['metadata']
            product = metadata.get('product', 'Unknown')
            issue = metadata.get('issue', 'Unknown')
            
            context_parts.append(
                f"[Source {i}] Product: {product}, Issue: {issue}\n"
                f"Complaint excerpt: {chunk_text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt template
        prompt = f"""You are a financial analyst assistant for CrediTrust Financial, a digital finance company serving East African markets. Your task is to answer questions about customer complaints based on the retrieved complaint excerpts provided below.

IMPORTANT INSTRUCTIONS:
- Use ONLY the information provided in the context below to answer the question
- If the context doesn't contain enough information to answer the question, clearly state that you don't have enough information
- Be concise but informative
- Focus on actionable insights that would help product managers, support teams, and compliance officers
- If multiple complaints mention similar issues, synthesize them into a coherent answer
- Cite which source(s) you're using when relevant (e.g., "According to Source 1 and Source 3...")

CONTEXT (Retrieved Complaint Excerpts):
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]] = None,
        use_llm: bool = True,
        model_name: str = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using RAG pipeline.
        
        Args:
            question: User's question
            retrieved_chunks: Pre-retrieved chunks (if None, will retrieve)
            use_llm: Whether to use LLM or return template response
            model_name: Name of LLM model to use
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve if not provided
        if retrieved_chunks is None:
            retrieved_chunks = self.retrieve(question)
        
        # Create prompt
        prompt = self.create_prompt(question, retrieved_chunks)
        
        # Generate answer
        if use_llm:
            answer = self._generate_with_llm(prompt, model_name)
        else:
            # Template response for testing
            answer = f"Based on the retrieved complaints, I found {len(retrieved_chunks)} relevant excerpts. [LLM generation would go here]"
        
        # Format sources
        sources = []
        for chunk_info in retrieved_chunks:
            sources.append({
                'text': chunk_info['chunk_text'][:200] + '...' if len(chunk_info['chunk_text']) > 200 else chunk_info['chunk_text'],
                'product': chunk_info['metadata'].get('product', 'Unknown'),
                'issue': chunk_info['metadata'].get('issue', 'Unknown'),
                'similarity': chunk_info['similarity_score']
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources': len(retrieved_chunks),
            'prompt': prompt
        }
    
    def _generate_with_llm(self, prompt: str, model_name: str = None) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Formatted prompt
            model_name: Optional model name override
            
        Returns:
            Generated answer string
        """
        # Try to use available LLM backends
        # Priority: HuggingFace Transformers > LangChain > Template
        
        try:
            # Try HuggingFace transformers first
            from transformers import pipeline
            
            if not self.llm_initialized:
                if model_name is None:
                    # Use a small, fast model for local inference
                    model_name = "microsoft/DialoGPT-medium"  # Fallback
                    # Or use: "google/flan-t5-base" for faster inference
                
                print(f"Initializing LLM: {model_name}")
                try:
                    self.llm = pipeline(
                        "text-generation",
                        model=model_name,
                        max_length=512,
                        do_sample=True,
                        temperature=0.7
                    )
                    self.llm_initialized = True
                except Exception as e:
                    print(f"Could not load {model_name}: {e}")
                    print("Falling back to template response...")
                    return self._template_response(prompt)
            
            # Generate answer
            response = self.llm(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
            
            # Extract generated text
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')
                # Remove the prompt from the beginning
                if prompt in generated_text:
                    answer = generated_text.replace(prompt, '').strip()
                else:
                    answer = generated_text.strip()
                return answer
            else:
                return self._template_response(prompt)
                
        except ImportError:
            print("transformers library not available. Using template response.")
            return self._template_response(prompt)
        except Exception as e:
            print(f"Error generating with LLM: {e}")
            return self._template_response(prompt)
    
    def _template_response(self, prompt: str) -> str:
        """Generate a template-based response when LLM is not available."""
        return """Based on the retrieved complaint excerpts, I can provide the following insights:

[This is a template response. To get actual LLM-generated answers, please install transformers library and configure an LLM model. For production use, consider using OpenAI API, Anthropic Claude, or other cloud-based LLM services.]

The retrieved complaints show relevant information that would be used to generate a comprehensive answer to your question."""
    
    def query(self, question: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query pipeline.
        
        Args:
            question: User's question
            use_llm: Whether to use LLM for generation
            
        Returns:
            Complete response dictionary
        """
        return self.generate_answer(question, use_llm=use_llm)
