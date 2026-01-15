"""
Task 4: Interactive Chat Interface for RAG Complaint Chatbot

This is the main application interface using Gradio.
"""

import gradio as gr
from pathlib import Path
import sys
from typing import Tuple, List
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline


class ComplaintChatbotApp:
    """
    Gradio-based chat interface for the complaint analysis chatbot.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.rag_pipeline = None
        self.initialized = False
        
    def initialize_pipeline(self):
        """Initialize the RAG pipeline (lazy loading)."""
        if not self.initialized:
            try:
                print("Initializing RAG Pipeline...")
                self.rag_pipeline = RAGPipeline(
                    vector_store_dir=Path('vector_store'),
                    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                    top_k=5
                )
                self.initialized = True
                print("✓ RAG Pipeline initialized successfully")
                return True, "Pipeline initialized successfully!"
            except FileNotFoundError as e:
                error_msg = (
                    f"Error loading vector store: {str(e)}\n\n"
                    "Please ensure:\n"
                    "1. Pre-built embeddings exist at: data/data/complaint_embeddings-002.parquet\n"
                    "2. Or vector store files exist in: vector_store/\n\n"
                    "If you haven't created the vector store yet, please run Task 2 first."
                )
                return False, error_msg
            except Exception as e:
                error_msg = f"Error initializing pipeline: {str(e)}"
                return False, error_msg
        return True, "Pipeline already initialized"
    
    def format_sources(self, sources: List[dict]) -> str:
        """
        Format source chunks for display.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Formatted string with sources
        """
        if not sources:
            return "No sources retrieved."
        
        formatted = "**Retrieved Sources:**\n\n"
        for i, src in enumerate(sources[:3], 1):  # Show top 3 sources
            product = src.get('product', 'Unknown')
            issue = src.get('issue', 'Unknown')
            similarity = src.get('similarity', 0)
            text = src.get('text', '')
            
            formatted += f"**Source {i}** (Similarity: {similarity:.3f})\n"
            formatted += f"- Product: {product}\n"
            formatted += f"- Issue: {issue}\n"
            formatted += f"- Excerpt: {text}\n\n"
        
        if len(sources) > 3:
            formatted += f"*... and {len(sources) - 3} more sources*\n"
        
        return formatted
    
    def query_chatbot(
        self,
        question: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        """
        Process a user question and return the answer.
        
        Args:
            question: User's question
            history: Chat history
            
        Returns:
            Tuple of (answer, sources, updated_history)
        """
        if not question or question.strip() == "":
            return "", "", history
        
        # Initialize pipeline if needed
        if not self.initialized:
            success, msg = self.initialize_pipeline()
            if not success:
                return f"Error: {msg}", "", history
        
        try:
            # Get response from RAG pipeline
            response = self.rag_pipeline.query(question, use_llm=True)
            
            answer = response['answer']
            sources = response['sources']
            
            # Format sources for display
            sources_display = self.format_sources(sources)
            
            # Update history
            history.append((question, answer))
            
            return answer, sources_display, history
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            return error_msg, "", history
    
    def clear_chat(self) -> Tuple[str, str, List]:
        """Clear the chat history."""
        return "", "", []
    
    def create_interface(self):
        """Create and return the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
        
        with gr.Blocks(css=css, theme=gr.themes.Soft()) as app:
            # Header
            gr.Markdown(
                """
                # 🏦 CrediTrust Financial - Complaint Analysis Chatbot
                
                **Intelligent Complaint Analysis for Financial Services**
                
                Ask questions about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.
                """,
                elem_classes=["main-header"]
            )
            
            # Status message
            status = gr.Textbox(
                label="Status",
                value="Initializing pipeline...",
                interactive=False,
                visible=True
            )
            
            # Initialize pipeline on load
            app.load(
                fn=self.initialize_pipeline,
                inputs=[],
                outputs=[gr.Textbox(visible=False), status]
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=400,
                        show_label=True,
                        avatar_images=(None, "🏦"),
                        bubble_full_width=False
                    )
                    
                    # Question input
                    question_input = gr.Textbox(
                        label="Ask a question about customer complaints",
                        placeholder="e.g., Why are customers unhappy with Credit Cards?",
                        lines=2,
                        show_label=True
                    )
                    
                    # Buttons
                    with gr.Row():
                        submit_btn = gr.Button("Ask Question", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)
                
                with gr.Column(scale=1):
                    # Sources display
                    sources_display = gr.Markdown(
                        label="Retrieved Sources",
                        value="Sources will appear here after you ask a question.",
                        show_label=True
                    )
            
            # Example questions
            gr.Markdown("### 💡 Example Questions:")
            examples = gr.Examples(
                examples=[
                    "Why are customers unhappy with Credit Cards?",
                    "What are the main issues with Personal Loans?",
                    "What problems do customers face with Money Transfers?",
                    "What are the most common complaints about Savings Accounts?",
                    "Which product has the most billing disputes?",
                    "What are customers saying about transaction delays?",
                ],
                inputs=question_input
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                **About this tool:**
                - Uses Retrieval-Augmented Generation (RAG) to answer questions based on real customer complaints
                - Retrieves relevant complaint excerpts and generates insights
                - Sources are displayed for transparency and verification
                - Built for CrediTrust Financial internal teams (Product, Support, Compliance)
                """
            )
            
            # Event handlers
            question_input.submit(
                fn=self.query_chatbot,
                inputs=[question_input, chatbot],
                outputs=[gr.Textbox(visible=False), sources_display, chatbot]
            ).then(
                fn=lambda: "",  # Clear input after submission
                inputs=[],
                outputs=[question_input]
            )
            
            submit_btn.click(
                fn=self.query_chatbot,
                inputs=[question_input, chatbot],
                outputs=[gr.Textbox(visible=False), sources_display, chatbot]
            ).then(
                fn=lambda: "",  # Clear input after submission
                inputs=[],
                outputs=[question_input]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[question_input, sources_display, chatbot]
            )
        
        return app


def main():
    """Main function to launch the application."""
    print("Starting CrediTrust Complaint Analysis Chatbot...")
    
    app_instance = ComplaintChatbotApp()
    interface = app_instance.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public link
        show_error=True
    )


if __name__ == "__main__":
    main()
