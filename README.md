# 🚀 Advanced RAG Pipeline (Sentence Window + Auto-Merging)

# 🔍 Project Overview

- This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex, combining Sentence Window RAG and Auto-Merging RAG techniques with evaluation using TruLens.
- The goal is to build a highly accurate, context-aware, and grounded Question Answering system over documents while reducing hallucinations.

# ❓ WHAT DOES THIS PROJECT DO? (KYA KARTA HAI?) 🤔

- 📄 Takes documents (PDF/Text)
- ✂️ Splits them intelligently into context-aware chunks
- 🧠 Converts text into vector embeddings
- 🔍 Retrieves the most relevant context for a user query
- 🤖 Uses an LLM to generate accurate answers
- 📊 Evaluates answers using relevance & groundedness metrics

# ❓ WHAT IS THIS PROJECT? (YE PROJECT KYA HAI?) 🧩

- This is a production-ready RAG system designed for:
  - AI chatbots
  - Knowledge base Q&A
  - Research assistants
  - EdTech platforms
  - Enterprise document search
- It improves over basic RAG by using sentence-level context windows and hierarchical auto-merging retrieval.

# 🎯 Purpose of This Project

- ✅ Solve context loss in traditional chunking
- ✅ Reduce hallucinations in LLM responses
- ✅ Improve retrieval accuracy
- ✅ Provide evaluation metrics for RAG quality

# 🧱 Project Structure

    Advanced-RAG-Pipeline/
    │
    ├── RAG_Pipeline.ipynb
    ├── sentence_window_retrieval.ipynb
    ├── automerging_retrieval.ipynb
    ├── utils.py
    ├── eval_questions.txt
    │
    ├── sentence_index/
    ├── merging_index/
    └── data/

# ✨ Key Features

- 🪟 Sentence Window RAG
- 🔗 Auto-Merging Hierarchical Retrieval
- 🎯 Semantic Re-ranking
- 📊 TruLens-based Evaluation
- ⚡ OpenAI / Groq compatible
- 🧠 Context-aware answers

# 🐍 Python Version
- Python 3.10 (Recommended)

# 🧠 Sentence Window RAG – Algorithm Explained

## 🔍 Problem
- Traditional chunking breaks sentence context.

## ✅ Solution
- Each sentence is stored with its surrounding sentences (window).

## 🔁 Algorithm Steps
- Split document into sentences
- Create a window:
  - Previous sentence
  - Current sentence
  - Next sentence
- Generate embeddings for windows
- Retrieve top-k similar windows
- Replace metadata with original context
- Re-rank results semantically
- 🪟 This preserves meaning and improves accuracy.

# 🔗 Auto-Merging RAG – Algorithm Overview

## 🔍 Problem
- Small chunks lose context, large chunks add noise.

## ✅ Solution
- Hierarchical chunking with automatic merging.

## 🔁 Algorithm Steps
- Chunk document into multiple levels (2048 → 512 → 128)
- Build parent-child relationships
- Index only leaf nodes
- Retrieve relevant leaf nodes
- Automatically merge parent context
- Pass merged context to LLM

# 📥 Input Process

- 📄 Documents (PDF / TXT)
- ❓ User Query
- 🧠 LLM (OpenAI / Groq)
- 🔢 Embedding Model

### Example:
- What is the installation process?

# 📤 Output Process

- ✅ Generated Answer
- 📚 Retrieved Context
- 📊 Evaluation Metrics

### Example:
- Answer Relevance: 0.91
- Context Relevance: 0.88
- Groundedness: 0.90

# 🔄 Flow Diagram (High-Level)

    Documents
       ↓
    Chunking (Sentence / Hierarchical)
       ↓
    Embeddings
       ↓
    Vector Index
       ↓
    Retriever + Re-ranker
       ↓
    Merged Context
       ↓
    LLM Answer
       ↓
    TruLens Evaluation

# 🧪 Evaluation Metrics (TruLens)

- 🎯 Answer Relevance
- 📚 Context Relevance
- 🧠 Groundedness (Hallucination check)

# 🚀 Real-World Use Cases

- 📖 AI Tutor Systems
- 🏢 Enterprise Knowledge Chatbots
- 📑 Research Paper Assistants
- 🎓 EdTech Platforms
- 🤝 Customer Support AI

# 👨‍💻 Built By (Professional)

## Built by: Sarfraz Khan
- Role: AI / ML Engineer | Data Scientist 
