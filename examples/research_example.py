import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent import ResearchAgent

def main():
    """
    Example of using the ResearchAgent with a sample document.
    """
    parser = argparse.ArgumentParser(description="Research Agent Example")
    parser.add_argument("--document", type=str, required=True, help="Path to document (PDF or text)")
    args = parser.parse_args()
    
    document_path = args.document
    
    print(f"Initializing Research Agent...")
    agent = ResearchAgent()
    
    # Process document based on extension
    if document_path.lower().endswith('.pdf'):
        print(f"Processing PDF: {document_path}")
        agent.process_pdf(document_path)
    elif document_path.lower().endswith(('.txt', '.md')):
        print(f"Processing text file: {document_path}")
        agent.process_text_file(document_path)
    else:
        print(f"Unsupported file format: {document_path}")
        sys.exit(1)
    
    # Build vector database
    print("Building vector database...")
    agent.build_vector_database()
    
    # Optional: Save the vector database
    # agent.save_vector_database("./vector_db")
    
    print("\nResearch Agent is ready!")
    print("You can ask questions about the document. Type 'exit' to quit.")
    
    # Question answering loop
    while True:
        user_input = input("\nEnter your question: ")
        
        if user_input.lower() in ("exit", "quit", "q"):
            break
        
        # Get answer
        result = agent.answer_question(user_input)
        
        # Display results
        print("\n" + "="*50)
        print("ANSWER:")
        print(result["answer"])
        print("-"*50)
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Relevant chunks: {result['num_relevant_chunks']}")
        print("="*50)

if __name__ == "__main__":
    main()