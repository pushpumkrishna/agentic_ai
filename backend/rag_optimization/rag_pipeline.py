# --- LangChain and LLM Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq

# --- Document Loading and Vector Store ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Prompting and Document Utilities ---
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# --- Core and Output Parsers ---
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod

# --- LangGraph for Workflow Graphs ---
from langgraph.graph import END, StateGraph

# --- Standard Library Imports ---
from time import monotonic
from dotenv import load_dotenv
from pprint import pprint
import os

# --- Datasets and Typing ---
from datasets import Dataset
from typing_extensions import TypedDict
from IPython.display import display, Image
from typing import List, TypedDict
from typing import TypedDict, List, Dict

# --- RAGAS Metrics for Evaluation ---
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity
)

import langgraph

from backend.config.azure_models import AzureOpenAIModels
from backend.config.logging_lib import logger
from backend.rag_optimization.build_graph import GraphRetrieval
from backend.rag_optimization.encoding import EncodeEmbeddings
from backend.rag_optimization.retrieve_data import RetrieveData
from backend.rag_optimization.step_1_preprocessing import ProcessDocument
# --- Helper Functions ---


# --- Load environment variables (e.g., API keys) ---
load_dotenv(override=True)

# --- Set environment variable for debugging (optional) ---
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"
hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"

# Retrieve the Groq API key from environment variable (for use by Groq LLMs)
groq_api_key = os.getenv('GROQ_API_KEY')

handler = ProcessDocument(hp_pdf_path)
chapter_summaries, book_quotes_list = handler.preprocess_pipeline()
for index, value in enumerate(chapter_summaries):
    print(value.metadata, value.page_content)
    print("book_quotes_list:: ", book_quotes_list)

encoding_handler = EncodeEmbeddings()
chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore = (
    encoding_handler.create_vector_db(chapter_summaries, book_quotes_list, hp_pdf_path))
logger.info(f"Vector ")

retriever_handler = RetrieveData(chunks_vector_store=chunks_vector_store,
                                 chapter_summaries_vector_store=chapter_summaries_vector_store,
                                 book_quotes_vectorstore=book_quotes_vectorstore)

result = (
    retriever_handler.run_retriever_pipeline())
logger.info(f"Retriever completed... ")

GraphRetrieval().graph_pipeline()



#
# def encode_book(path, chunk_size=1000, chunk_overlap=200):
#     """
#     Encodes a PDF book into a FAISS vector store using OpenAI embeddings.
#
#     Args:
#         path (str): The path to the PDF file.
#         chunk_size (int): The desired size of each text chunk.
#         chunk_overlap (int): The amount of overlap between consecutive chunks.
#
#     Returns:
#         FAISS: A FAISS vector store containing the encoded book content.
#     """
#
#     # 1. Load the PDF document using PyPDFLoader
#     loader = PyPDFLoader(path)
#     documents = loader.load()
#
#     # 2. Split the document into chunks for embedding
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len
#     )
#     texts = text_splitter.split_documents(documents)
#
#     # 3. Clean up the text chunks (replace unwanted characters)
#     cleaned_texts = replace_t_with_space(texts)
#
#     # 4. Create OpenAI embeddings and encode the cleaned text chunks into a FAISS vector store
#     embeddings = HuggingFaceEmbeddings(
#             model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
#             model_kwargs={
#                 "device": "cpu",
#             }
#         )
#     vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
#
#     # 5. Return the vector store
#     return vectorstore
#
#
# """Encoding Chapter Summaries into Vector Store"""
# def encode_chapter_summaries(chapter_summaries):
#     """
#     Encodes a list of chapter summaries into a FAISS vector store using OpenAI embeddings.
#
#     Args:
#         chapter_summaries (list): A list of Document objects representing the chapter summaries.
#
#     Returns:
#         FAISS: A FAISS vector store containing the encoded chapter summaries.
#     """
#     # Create OpenAI embeddings instance
#     embeddings = HuggingFaceEmbeddings(
#         model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
#         model_kwargs={
#             "device": "cpu",
#         }
#     )
#
#     # Encode the chapter summaries into a FAISS vector store
#     chapter_summaries_vectorstore = FAISS.from_documents(chapter_summaries, embeddings)
#
#     # Return the vector store
#     return chapter_summaries_vectorstore
#
#
# """Encoding Quotes into Vector Store"""
# def encode_quotes(book_quotes_list):
#     """
#     Encodes a list of book quotes into a FAISS vector store using OpenAI embeddings.
#
#     Args:
#         book_quotes_list (list): A list of Document objects, each representing a quote from the book.
#
#     Returns:
#         FAISS: A FAISS vector store containing the encoded book quotes.
#     """
#     # Create OpenAI embeddings instance
#     embeddings = HuggingFaceEmbeddings(
#         model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
#         model_kwargs={
#             "device": "cpu",
#         }
#     )
#
#     # Encode the book quotes into a FAISS vector store
#     quotes_vectorstore = FAISS.from_documents(book_quotes_list, embeddings)
#
#     # Return the vector store
#     return quotes_vectorstore
#
#
# """Creating Vector Stores and Retrievers for Book and Chapter Summaries"""
# # --- Create or Load Vector Stores for Book Chunks, Chapter Summaries, and Book Quotes ---
#
# # Check if the vector stores already exist on disk
# if (
#         os.path.exists("chunks_vector_store") and
#         os.path.exists("chapter_summaries_vector_store") and
#         os.path.exists("book_quotes_vectorstore")
# ):
#     # If vector stores exist, load them using OpenAI embeddings
#     embeddings = OpenAIEmbeddings()
#     chunks_vector_store = FAISS.load_local(
#         "chunks_vector_store", embeddings, allow_dangerous_deserialization=True
#     )
#     chapter_summaries_vector_store = FAISS.load_local(
#         "chapter_summaries_vector_store", embeddings, allow_dangerous_deserialization=True
#     )
#     book_quotes_vectorstore = FAISS.load_local(
#         "book_quotes_vectorstore", embeddings, allow_dangerous_deserialization=True
#     )
# else:
#     # If vector stores do not exist, encode and save them
#
#     # 1. Encode the book into a vector store of chunks
#     chunks_vector_store = encode_book(hp_pdf_path, chunk_size=1000, chunk_overlap=200)
#
#     # 2. Encode the chapter summaries into a vector store
#     chapter_summaries_vector_store = encode_chapter_summaries(chapter_summaries)
#
#     # 3. Encode the book quotes into a vector store
#     book_quotes_vectorstore = encode_quotes(book_quotes_list)
#
#     # 4. Save the vector stores to disk for future use
#     chunks_vector_store.save_local("chunks_vector_store")
#     chapter_summaries_vector_store.save_local("chapter_summaries_vector_store")
#     book_quotes_vectorstore.save_local("book_quotes_vectorstore")
