# RAG-with-Mistral-and-LangChain-LlamaIndex-Haystack
This repository contains implementations of Retrieval-Augmented Generation (RAG) systems using multiple frameworks, all powered by Mistral AI. The system is designed to retrieve relevant information from text documents and generate accurate answers to user queries.
# Document QA System with Mistral AI

This repository contains implementations of Retrieval-Augmented Generation (RAG) systems using multiple frameworks, all powered by Mistral AI. The system is designed to retrieve relevant information from text documents and generate accurate answers to user queries.

## 🌟 Features

- **Multiple Framework Implementations**: Choose between LangChain, LlamaIndex, and Haystack
- **Mistral AI Integration**: Leveraging Mistral's powerful embedding and language models
- **Document Preprocessing**: Automatic chunking and vectorization of text data
- **Semantic Search**: Find relevant document chunks based on query similarity
- **Context-Aware Responses**: Generate responses that consider document context

## 📋 Implementations

This repository showcases three different implementations of a RAG system:

### 1. Raw Implementation with FAISS

A basic implementation using Mistral AI's embeddings with FAISS for vector similarity search.

```python
# Raw implementation with FAISS and Mistral AI
import faiss
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Create embeddings, build index, and search for relevant chunks
```

### 2. LangChain Implementation

Using LangChain's components to build a retrieval chain.

```python
# LangChain implementation
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import MistralAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
```

### 3. LlamaIndex Implementation

Using LlamaIndex for document indexing and querying.

```python
# LlamaIndex implementation
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import MistralAI
from llama_index.embeddings import MistralAIEmbedding
from llama_index import Settings
```

### 4. Haystack Implementation

Using Haystack's pipeline approach.

```python
# Haystack implementation
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.mistral import MistralDocumentEmbedder, MistralTextEmbedder
from haystack_integrations.components.generators.mistral import MistralChatGenerator
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-qa-mistral.git
cd document-qa-mistral
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Prepare your text document (default: `essay.txt`)

2. Set your Mistral API key:
```python
import os
os.environ["MISTRAL_API_KEY"] = "your_api_key_here"
# Or use getpass for interactive input
from getpass import getpass
api_key = getpass("Type your API Key: ")
```

3. Choose your implementation and run the corresponding script:
```bash
# For raw implementation
python raw_implementation.py

# For LangChain
python langchain_implementation.py

# For LlamaIndex
python llamaindex_implementation.py

# For Haystack
python haystack_implementation.py
```

## 📝 Example Query

All implementations use the same example query:
```
"What were the two main things the author worked on before college?"
```

## 📦 Requirements

- mistralai
- faiss-cpu==1.7.4
- langchain
- llama-index
- haystack-ai
- haystack-integrations[mistral]
- requests

## 📚 Project Structure

```
document-qa-mistral/
├── raw_implementation.py        # Raw implementation with FAISS
├── langchain_implementation.py  # LangChain implementation
├── llamaindex_implementation.py # LlamaIndex implementation
├── haystack_implementation.py   # Haystack implementation
├── requirements.txt             # Project dependencies
├── essay.txt                    # Example text document
└── README.md                    # This file
```

## 🤔 Implementation Comparison

| Framework | Pros | Cons |
|-----------|------|------|
| Raw/FAISS | - Full control<br>- Minimal dependencies<br>- Lightweight | - More code to write<br>- No built-in abstractions |
| LangChain | - Well-established ecosystem<br>- Many built-in tools<br>- Good documentation | - More dependencies<br>- Sometimes complex |
| LlamaIndex | - Document-focused<br>- Simple API<br>- Good for structured data | - More specialized<br>- Fewer customization options |
| Haystack | - Pipeline approach<br>- Modular components<br>- Good for production | - Steeper learning curve<br>- More verbose |

## 🔍 How It Works

1. **Document Processing**:
   - Load text documents
   - Split into manageable chunks
   - Generate embeddings for each chunk

2. **Query Processing**:
   - Generate embeddings for user query
   - Find most similar document chunks
   - Combine query with relevant contexts

3. **Response Generation**:
   - Pass combined context to Mistral AI's language model
   - Generate concise, accurate responses

## 🔮 Future Improvements

- Add support for more document types (PDF, DOCX, etc.)
- Implement hybrid search (combining sparse and dense retrievers)
- Add evaluation metrics for response quality
- Create a simple web interface
- Implement streaming responses

## 📄 License

MIT

## 🙏 Acknowledgements

- [Mistral AI](https://mistral.ai/) for their powerful language and embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [LangChain](https://github.com/langchain-ai/langchain), [LlamaIndex](https://github.com/run-llama/llama_index), and [Haystack](https://github.com/deepset-ai/haystack) for their excellent frameworks
