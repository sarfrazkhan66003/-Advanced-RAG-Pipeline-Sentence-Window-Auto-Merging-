"""
RAG Helper Functions
====================
This module provides helper functions for loading and processing documents
for RAG (Retrieval-Augmented Generation) pipelines using llama_index.

Functions:
- load_pdf_documents: Load PDF documents from file path(s)
- merge_documents: Merge multiple documents into a single Document
- get_merged_document: Load and merge PDFs in one step
"""

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PyMuPDFReader


def load_pdf_documents(file_paths):
    """
    Load PDF documents from one or more file paths.
    
    Args:
        file_paths (str or list): Single file path or list of file paths to PDF files
        
    Returns:
        list: List of Document objects
        
    Example:
        # Single file
        docs = load_pdf_documents("path/to/file.pdf")
        
        # Multiple files
        docs = load_pdf_documents(["file1.pdf", "file2.pdf"])
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    documents = SimpleDirectoryReader(
        input_files=file_paths,
        file_extractor={".pdf": PyMuPDFReader()}
    ).load_data()
    
    return documents


def merge_documents(documents):
    """
    Merge multiple documents into a single Document object.
    
    Args:
        documents (list): List of Document objects to merge
        
    Returns:
        Document: Single merged Document object
        
    Example:
        merged_doc = merge_documents(documents)
    """
    merged_text = '\n\n'.join([doc.text for doc in documents])
    return Document(text=merged_text)


def get_merged_document(file_paths):
    """
    Load PDF documents and merge them into a single Document.
    This is a convenience function that combines load_pdf_documents and merge_documents.
    
    Args:
        file_paths (str or list): Single file path or list of file paths to PDF files
        
    Returns:
        Document: Single merged Document object
        
    Example:
        # For a single PDF
        document = get_merged_document(r"C:\path\to\file.pdf")
        
        # For multiple PDFs
        document = get_merged_document([r"C:\path\to\file1.pdf", r"C:\path\to\file2.pdf"])
    """
    documents = load_pdf_documents(file_paths)
    return merge_documents(documents)


# Example usage
if __name__ == "__main__":
    # Example: Load a single PDF
    pdf_path = r"C:\Users\DELL\OneDrive\Desktop\Sarfraz\PW Data Science\Project\RAG_Application_Using_LLM\Medical_Cost_Prediction.pdf"
    
    # Load documents
    documents = load_pdf_documents(pdf_path)
    print(f"Loaded {len(documents)} documents")
    
    # Merge into single document
    merged_doc = merge_documents(documents)
    print(f"Merged document length: {len(merged_doc.text)} characters")
    
    # Or use the convenience function
    document = get_merged_document(pdf_path)
    print(f"Document loaded and merged: {len(document.text)} characters")
