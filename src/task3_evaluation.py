"""
Task 3: RAG Pipeline Evaluation

This script evaluates the RAG pipeline with representative questions.
"""

import pandas as pd
from pathlib import Path
from src.rag_pipeline import RAGPipeline
import json


def evaluate_rag_pipeline(
    rag_pipeline: RAGPipeline,
    evaluation_questions: list,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Evaluate the RAG pipeline with a set of questions.
    
    Args:
        rag_pipeline: Initialized RAGPipeline instance
        evaluation_questions: List of question strings
        output_path: Optional path to save evaluation results
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    print("="*80)
    print("RAG Pipeline Evaluation")
    print("="*80)
    
    for i, question in enumerate(evaluation_questions, 1):
        print(f"\n[{i}/{len(evaluation_questions)}] Processing question: {question}")
        print("-" * 80)
        
        # Run RAG pipeline
        response = rag_pipeline.query(question, use_llm=True)
        
        # Extract information
        answer = response['answer']
        sources = response['sources']
        num_sources = response['num_sources']
        
        # Display top 2 sources
        top_sources = sources[:2]
        source_texts = []
        for j, src in enumerate(top_sources, 1):
            source_texts.append(f"Source {j}: [{src['product']}] {src['text']}")
        
        sources_display = "\n".join(source_texts)
        
        # For evaluation, you would manually score each answer
        # This is a placeholder - you should review and score each answer
        quality_score = None  # To be filled manually: 1-5 scale
        
        result = {
            'Question': question,
            'Generated Answer': answer[:500] + '...' if len(answer) > 500 else answer,
            'Retrieved Sources': sources_display,
            'Number of Sources': num_sources,
            'Quality Score': quality_score,
            'Comments/Analysis': 'To be filled after manual review'
        }
        
        results.append(result)
        
        # Print summary
        print(f"Answer length: {len(answer)} characters")
        print(f"Retrieved {num_sources} sources")
        print(f"Top source similarity: {sources[0]['similarity']:.4f}" if sources else "N/A")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df_results.to_csv(output_path.with_suffix('.csv'), index=False)
        
        # Save as Markdown table
        markdown_table = df_results.to_markdown(index=False)
        with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            f.write("# RAG Pipeline Evaluation Results\n\n")
            f.write(markdown_table)
        
        print(f"\n✓ Evaluation results saved to: {output_path}")
    
    return df_results


def main():
    """Main evaluation function."""
    
    # Define evaluation questions
    evaluation_questions = [
        "Why are customers unhappy with Credit Cards?",
        "What are the main issues with Personal Loans?",
        "What problems do customers face with Money Transfers?",
        "What are the most common complaints about Savings Accounts?",
        "Which product has the most billing disputes?",
        "What are customers saying about transaction delays?",
        "Are there any fraud-related complaints?",
        "What issues are customers reporting with account access?",
        "What are the top complaints across all products?",
        "How do complaints differ between Credit Cards and Personal Loans?"
    ]
    
    print("Initializing RAG Pipeline...")
    
    # Initialize RAG pipeline
    try:
        rag_pipeline = RAGPipeline(
            vector_store_dir=Path('../vector_store'),
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            top_k=5
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. Pre-built embeddings file exists at: data/data/complaint_embeddings-002.parquet")
        print("2. Or vector store files exist in: vector_store/")
        return
    
    # Run evaluation
    output_path = Path('../notebooks/evaluation_results')
    df_results = evaluate_rag_pipeline(
        rag_pipeline=rag_pipeline,
        evaluation_questions=evaluation_questions,
        output_path=output_path
    )
    
    # Display summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    print(f"Total questions evaluated: {len(evaluation_questions)}")
    print(f"Average sources per question: {df_results['Number of Sources'].mean():.2f}")
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("1. Review each answer manually")
    print("2. Assign quality scores (1-5) based on:")
    print("   - Relevance to question")
    print("   - Use of retrieved context")
    print("   - Completeness of answer")
    print("   - Clarity and coherence")
    print("3. Add comments/analysis for each question")
    print("4. Update the evaluation table in your report")


if __name__ == "__main__":
    main()
