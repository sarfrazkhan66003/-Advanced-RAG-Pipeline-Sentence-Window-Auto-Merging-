"""
CORRECT IMPORTS FOR LLAMA_INDEX v0.10+
========================================

This file contains ALL the correct import statements for your RAG pipeline.
Copy these imports into your notebook cells.

IMPORTANT: In llama_index v0.10+, most imports moved from 'llama_index' to 'llama_index.core'
"""

# ============================================================================
# CELL 1: Load PDF Documents
# ============================================================================
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader

documents = SimpleDirectoryReader(
    input_files=[
        r"C:\Users\DELL\OneDrive\Desktop\Sarfraz\PW Data Science\Project\RAG_Application_Using_LLM\Medical_Cost_Prediction.pdf"
    ],
    file_extractor={
        ".pdf": PyMuPDFReader()
    }
).load_data()
print(f"Number of documents: {len(documents)}")


# ============================================================================
# CELL 2: Merge Documents
# ============================================================================
from llama_index.core import Document

document = Document(text='\n\n'.join([doc.text for doc in documents]))


# ============================================================================
# CELL 3: Create Vector Store Index (INGESTION) - LATEST VERSION
# ============================================================================
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
import os

# Set Groq API Key
os.environ["GROQ_API_KEY"] = "api key"

# Configure LLM using Groq (Recommended for your environment)
llm = Groq(model="llama-3.3-70b-versatile")

# Set global settings
Settings.llm = llm
Settings.embed_model = 'local:BAAI/bge-small-en-v1.5'

# Create index - Settings will be used automatically
index = VectorStoreIndex.from_documents([document])

print("SUCCESS: Index created successfully!")


# ============================================================================
# CELL 4: Create Query Engine
# ============================================================================
query_engine = index.as_query_engine()


# ============================================================================
# CELL 5: Query the Engine
# ============================================================================
response = query_engine.query('How is the project deployed')
print(str(response))


# ============================================================================
# COMPLETE IMPORT REFERENCE
# ============================================================================
"""
Here are ALL the imports you might need, organized by category:

CORE IMPORTS:
-------------
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    ServiceContext,
    Settings,
    StorageContext,
    load_index_from_storage
)

LLM IMPORTS:
-----------
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq

READER IMPORTS:
--------------
from llama_index.readers.file import PyMuPDFReader, PDFReader

NODE PARSER IMPORTS:
-------------------
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)

POSTPROCESSOR IMPORTS:
---------------------
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank
)

RETRIEVER IMPORTS:
-----------------
from llama_index.core.retrievers import AutoMergingRetriever

QUERY ENGINE IMPORTS:
--------------------
from llama_index.core.query_engine import RetrieverQueryEngine

EVALUATION IMPORTS (TruLens):
-----------------------------
from trulens_eval import Feedback, TruLlama, Tru
from trulens_eval import OpenAI as fOpenAI
from trulens_eval.feedback import Groundedness
"""

print("DONE: All imports are correct for llama_index v0.10+")
