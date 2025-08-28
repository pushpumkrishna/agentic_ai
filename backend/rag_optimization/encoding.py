import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.constants import chapter_summaries


class EncodeEmbeddings:

    def encode_book(self, path, chunk_size=1000, chunk_overlap=200):
        """
        Encodes a PDF book into a FAISS vector store using OpenAI embeddings.

        Args:
            path (str): The path to the PDF file.
            chunk_size (int): The desired size of each text chunk.
            chunk_overlap (int): The amount of overlap between consecutive chunks.

        Returns:
            FAISS: A FAISS vector store containing the encoded book content.
        """

        # 1. Load the PDF document using PyPDFLoader
        loader = PyPDFLoader(path)
        documents = loader.load()

        # 2. Split the document into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # 3. Clean up the text chunks (replace unwanted characters)
        cleaned_texts = self.replace_t_with_space(texts)

        # 4. Create OpenAI embeddings and encode the cleaned text chunks into a FAISS vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
            }
        )
        vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

        # 5. Return the vector store
        return vectorstore

    @staticmethod
    def replace_t_with_space(list_of_documents):
        """
        Replaces all tab characters ('\t') with spaces in the page content of each document.

        Args:
            list_of_documents (list): A list of document objects, each with a 'page_content' attribute.

        Returns:
            list: The modified list of documents with tab characters replaced by spaces.
        """
        for doc in list_of_documents:
            doc.page_content = doc.page_content.replace('\t', ' ')
        return list_of_documents

    @staticmethod
    def encode_chapter_summaries(chapter_summaries):
        """
        Encodes a list of chapter summaries into a FAISS vector store using OpenAI embeddings.
        Encoding Chapter Summaries into Vector Store
        Args:
            chapter_summaries (list): A list of Document objects representing the chapter summaries.

        Returns:
            FAISS: A FAISS vector store containing the encoded chapter summaries.
        """
        # Create OpenAI embeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
            }
        )
        # chapter_summaries is a list of strings
        # documents = [Document(page_content=summary) for summary in chapter_summaries]

        documents = chapter_summaries

        # Encode the chapter summaries into a FAISS vector store
        chapter_summaries_vectorstore = FAISS.from_documents(documents, embeddings)

        # Return the vector store
        return chapter_summaries_vectorstore

    @staticmethod
    def encode_quotes(book_quotes_list):
        """
        Encodes a list of book quotes into a FAISS vector store using OpenAI embeddings.
        Encoding Quotes into Vector Store
        Args:
            book_quotes_list (list): A list of Document objects, each representing a quote from the book.

        Returns:
            FAISS: A FAISS vector store containing the encoded book quotes.
        """
        # Create OpenAI embeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
            }
        )
        if len(book_quotes_list) > 0:
            # Encode the book quotes into a FAISS vector store
            quotes_vectorstore = FAISS.from_documents(book_quotes_list, embeddings)

            # Return the vector store
            return quotes_vectorstore
        return []

    def create_vector_db(self, chapter_summaries, book_quotes_list, hp_pdf_path):
        """Creating Vector Stores and Retrievers for Book and Chapter Summaries
        --- Create or Load Vector Stores for Book Chunks, Chapter Summaries, and Book Quotes ---
        """

        # Check if the vector stores already exist on disk
        if (
                os.path.exists("chunks_vector_store") and
                os.path.exists("chapter_summaries_vector_store") and
                os.path.exists("book_quotes_vectorstore")
        ):
            # If vector stores exist, load them using OpenAI embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
                model_kwargs={
                    "device": "cpu",
                }
            )
            chunks_vector_store = FAISS.load_local(
                "chunks_vector_store", embeddings, allow_dangerous_deserialization=True
            )
            chapter_summaries_vector_store = FAISS.load_local(
                "chapter_summaries_vector_store", embeddings, allow_dangerous_deserialization=True
            )
            book_quotes_vectorstore = FAISS.load_local(
                "book_quotes_vectorstore", embeddings, allow_dangerous_deserialization=True
            )
        else:
            # If vector stores do not exist, encode and save them

            # 1. Encode the book into a vector store of chunks
            chunks_vector_store = self.encode_book(hp_pdf_path, chunk_size=1000, chunk_overlap=200)

            # 2. Encode the chapter summaries into a vector store
            chapter_summaries_vector_store = self.encode_chapter_summaries(chapter_summaries)

            # 3. Encode the book quotes into a vector store
            book_quotes_vectorstore = self.encode_quotes(book_quotes_list)

            # 4. Save the vector stores to disk for future use
            if isinstance(chunks_vector_store, FAISS):
                chunks_vector_store.save_local("chunks_vector_store")
            if isinstance(chapter_summaries_vector_store, FAISS):
                chapter_summaries_vector_store.save_local("chapter_summaries_vector_store")
            if isinstance(book_quotes_vectorstore, FAISS):
                book_quotes_vectorstore.save_local("book_quotes_vectorstore")

        return chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore


if __name__ == "__main__":
    handler = EncodeEmbeddings()
    hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
    A, B, C = handler.create_vector_db(chapter_summaries, [], hp_pdf_path)
    print((A, B, C))
    print("done")
