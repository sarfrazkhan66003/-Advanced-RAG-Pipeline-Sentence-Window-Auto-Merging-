
import traceback
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader

try:
    print("Attempting to load PDF with PyMuPDFReader...")
    documents = SimpleDirectoryReader(
        input_files=[
            r"C:\Users\DELL\OneDrive\Desktop\Sarfraz\PW Data Science\Project\RAG_Application_Using_LLM\Medical_Cost_Prediction.pdf"
        ],
        file_extractor={
            ".pdf": PyMuPDFReader()
        },
        raise_on_error=True
    ).load_data()
    print(f"Number of documents: {len(documents)}")
except Exception:
    traceback.print_exc()
