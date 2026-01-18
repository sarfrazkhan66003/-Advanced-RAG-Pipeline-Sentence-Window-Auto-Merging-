
import traceback
try:
    from llama_index.readers.file import PyMuPDFReader
    print("PyMuPDFReader imported successfully from llama_index.readers.file")
except ImportError:
    print("PyMuPDFReader NOT found in llama_index.readers.file")
    try:
        from llama_index.readers.pymupdf import PyMuPDFReader
        print("PyMuPDFReader imported successfully from llama_index.readers.pymupdf")
    except ImportError:
        print("PyMuPDFReader NOT found in llama_index.readers.pymupdf")

from llama_index.core import SimpleDirectoryReader
try:
    # If we found it, use it. If not, this will fail or we need to define it.
    # Let's assume the previous try/except blocks helped us find where it is, 
    # but for this script execution we need to reference the right one.
    # I'll just try to use it if it's in local namespace, otherwise re-import.
    
    # Re-import logic for usage:
    try:
        from llama_index.readers.file import PyMuPDFReader
        reader = PyMuPDFReader()
    except ImportError:
        try:
             from llama_index.readers.pymupdf import PyMuPDFReader
             reader = PyMuPDFReader()
        except ImportError:
             print("Could not import PyMuPDFReader anywhere.")
             exit(1)

    print("Attempting to load PDF with PyMuPDFReader...")
    documents = SimpleDirectoryReader(
        input_files=[
            r"C:\Users\DELL\OneDrive\Desktop\Sarfraz\PW Data Science\Project\RAG_Application_Using_LLM\Medical_Cost_Prediction.pdf"
        ],
        file_extractor={
            ".pdf": reader
        }
    ).load_data()
    print(f"Number of documents: {len(documents)}")
except Exception:
    traceback.print_exc()
