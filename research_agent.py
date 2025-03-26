import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain.callbacks import get_openai_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ResearchAgent")

class ResearchAgent:
    """
    Research Agent that can extract information from documents and answer questions.
    
    This agent can:
    1. Process PDF and text documents
    2. Create vector embeddings for efficient retrieval
    3. Answer questions based on the document content
    4. Maintain conversation context
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.1,
        vector_db_path: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        self.vector_db_path = vector_db_path
        self.documents = []
        self.document_metadata = {}
        
        # Initialize embeddings model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize language model
        logger.info(f"Initializing language model: {llm_model_name}")
        self.llm = HuggingFaceHub(
            repo_id=llm_model_name,
            model_kwargs={"temperature": temperature}
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize QA chain
        self.chain = load_qa_chain(
            self.llm,
            chain_type="stuff",
            memory=self.memory
        )
        
        # Vector database
        self.vectordb = None
        
        logger.info("Research Agent initialized successfully")
    
    def process_pdf(self, pdf_path: str) -> None:
        """
        Extract text from a PDF file and add to documents.
        
        Args:
            pdf_path: Path to the PDF file
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            pdf_reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from each page
            for page in tqdm(pdf_reader.pages, desc="Extracting PDF text"):
                text += page.extract_text()
            
            # Store document metadata
            file_name = os.path.basename(pdf_path)
            self.document_metadata[file_name] = {
                "path": pdf_path,
                "type": "pdf",
                "pages": len(pdf_reader.pages)
            }
            
            # Split text into chunks
            self._split_and_add_text(text, source=file_name)
            
            logger.info(f"Successfully processed PDF: {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def process_text_file(self, text_path: str) -> None:
        """
        Extract text from a text file and add to documents.
        
        Args:
            text_path: Path to the text file
        """
        logger.info(f"Processing text file: {text_path}")
        
        if not os.path.exists(text_path):
            logger.error(f"Text file not found: {text_path}")
            raise FileNotFoundError(f"Text file not found: {text_path}")
        
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Store document metadata
            file_name = os.path.basename(text_path)
            self.document_metadata[file_name] = {
                "path": text_path,
                "type": "text"
            }
            
            # Split text into chunks
            self._split_and_add_text(text, source=file_name)
            
            logger.info(f"Successfully processed text file: {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def _split_and_add_text(self, text: str, source: str) -> None:
        """
        Split text into chunks and add to documents list.
        
        Args:
            text: Text to split
            source: Source identifier (filename)
        """
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Add chunks to documents list with metadata
        for i, chunk in enumerate(chunks):
            self.documents.append({
                "content": chunk,
                "metadata": {
                    "source": source,
                    "chunk": i
                }
            })
        
        logger.info(f"Added {len(chunks)} chunks from {source}")
    
    def build_vector_database(self) -> None:
        """
        Build vector database from processed documents.
        """
        if not self.documents:
            logger.warning("No documents processed. Vector database not built.")
            return
        
        logger.info(f"Building vector database with {len(self.documents)} chunks")
        
        try:
            # Extract text and metadata
            texts = [doc["content"] for doc in self.documents]
            metadatas = [doc["metadata"] for doc in self.documents]
            
            # Create vector database
            self.vectordb = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save vector database if path is provided
            if self.vector_db_path:
                self.save_vector_database(self.vector_db_path)
            
            logger.info("Vector database built successfully")
            
        except Exception as e:
            logger.error(f"Error building vector database: {e}")
            raise
    
    def save_vector_database(self, path: str) -> None:
        """
        Save vector database to disk.
        
        Args:
            path: Directory path to save vector database
        """
        if not self.vectordb:
            logger.warning("No vector database to save.")
            return
        
        os.makedirs(path, exist_ok=True)
        self.vectordb.save_local(path)
        logger.info(f"Vector database saved to: {path}")
    
    def load_vector_database(self, path: str) -> None:
        """
        Load vector database from disk.
        
        Args:
            path: Directory path to load vector database from
        """
        if not os.path.exists(path):
            logger.error(f"Vector database path not found: {path}")
            raise FileNotFoundError(f"Vector database path not found: {path}")
        
        self.vectordb = FAISS.load_local(path, self.embeddings)
        self.vector_db_path = path
        logger.info(f"Vector database loaded from: {path}")
    
    def answer_question(
        self, 
        question: str, 
        similarity_threshold: float = 0.7,
        max_docs: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question based on the document content.
        
        Args:
            question: The question to answer
            similarity_threshold: Threshold for similarity score
            max_docs: Maximum number of documents to retrieve
            
        Returns:
            Dictionary containing answer and metadata
        """
        if not self.vectordb:
            logger.error("Vector database not built. Process documents first.")
            raise ValueError("Vector database not built. Process documents first.")
        
        logger.info(f"Answering question: {question}")
        
        # Retrieve relevant documents
        docs = self.vectordb.similarity_search_with_score(
            question, 
            k=max_docs
        )
        
        # Filter by similarity threshold
        relevant_docs = []
        sources = set()
        
        for doc, score in docs:
            if score <= similarity_threshold:
                relevant_docs.append(doc)
                sources.add(doc.metadata.get("source", "unknown"))
        
        if not relevant_docs:
            logger.warning("No relevant documents found for the question")
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Get answer from language model
        with get_openai_callback() as cb:
            answer = self.chain.run(
                input_documents=relevant_docs,
                question=question
            )
        
        logger.info(f"Generated answer with {len(relevant_docs)} relevant chunks")
        
        return {
            "answer": answer,
            "sources": list(sources),
            "num_relevant_chunks": len(relevant_docs),
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens
        }
    
    def clear_memory(self) -> None:
        """
        Clear conversation memory.
        """
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_document_info(self) -> List[Dict[str, Any]]:
        """
        Get information about processed documents.
        
        Returns:
            List of dictionaries with document information
        """
        return [
            {
                "name": name,
                "path": info["path"],
                "type": info["type"],
                "pages": info.get("pages", None)
            }
            for name, info in self.document_metadata.items()
        ]

def main():
    """
    Main function to run the Research Agent from command line.
    """
    parser = argparse.ArgumentParser(description="Research Agent for document Q&A")
    parser.add_argument("--pdf", type=str, help="Path to PDF file")
    parser.add_argument("--text", type=str, help="Path to text file")
    parser.add_argument("--vector_db", type=str, help="Path to vector database")
    parser.add_argument("--save_vector_db", type=str, help="Path to save vector database")
    parser.add_argument("--question", type=str, help="Question to answer")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ResearchAgent()
    
    # Process documents or load vector database
    if args.pdf:
        agent.process_pdf(args.pdf)
        agent.build_vector_database()
    elif args.text:
        agent.process_text_file(args.text)
        agent.build_vector_database()
    elif args.vector_db:
        agent.load_vector_database(args.vector_db)
    else:
        logger.error("No input specified. Use --pdf, --text, or --vector_db")
        sys.exit(1)
    
    # Save vector database if requested
    if args.save_vector_db:
        agent.save_vector_database(args.save_vector_db)
    
    # Answer question if provided
    if args.question:
        result = agent.answer_question(args.question)
        print("\n" + "-"*50)
        print(f"Question: {args.question}")
        print("-"*50)
        print(f"Answer: {result['answer']}")
        print("-"*50)
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Relevant chunks: {result['num_relevant_chunks']}")
        print("-"*50 + "\n")
    else:
        # Interactive mode
        print("\nResearch Agent ready. Type 'exit' to quit.")
        print("Enter your question:")
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() in ("exit", "quit", "q"):
                break
            
            result = agent.answer_question(user_input)
            
            print("\n" + "-"*50)
            print(f"Answer: {result['answer']}")
            print("-"*50)
            print(f"Sources: {', '.join(result['sources'])}")
            print(f"Relevant chunks: {result['num_relevant_chunks']}")
            print("-"*50 + "\n")

if __name__ == "__main__":
    main()