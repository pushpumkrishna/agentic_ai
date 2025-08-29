import os
import asyncio
from typing import List, Tuple, Union, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config.logging_lib import logger


class EncodeEmbeddings:
    """
    Encoder helper class to create embedding vector stores for book chunks, chapter summaries, and quotes.
    """

    async def encode_book(
        self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> FAISS:
        """
        Description:
            Encodes a PDF book into a FAISS vector store using HuggingFace embeddings.
            Loads the PDF, splits into chunks and creates a FAISS vector store.

        Params:
            path (str): The path to the PDF file.
            chunk_size (int): The desired size of each text chunk.
            chunk_overlap (int): The amount of overlap between consecutive chunks.

        Return:
            FAISS: A FAISS vector store containing the encoded book content.

        Exceptions:
            TypeError: If path is not a string or chunk_size/chunk_overlap are not ints.
            FileNotFoundError: If the PDF at `path` does not exist.
            RuntimeError: If embedding or FAISS creation fails.
        """
        logger.info(f"Starting encode_book for path: {path}")

        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int):
            raise TypeError("chunk_size and chunk_overlap must be integers")
        if not os.path.exists(path):
            logger.error(f"PDF path does not exist: {path}")
            raise FileNotFoundError(path)

        try:
            # 1. Load the PDF document using PyPDFLoader (blocking -> to_thread)
            logger.info(f"Loading PDF using PyPDFLoader: {path}")
            loader = PyPDFLoader(path)
            documents: List[Document] = await asyncio.to_thread(loader.load)

            # 2. Split the document into chunks for embedding (blocking -> to_thread)
            logger.info(
                f"Splitting documents into chunks: ({chunk_size}, {chunk_overlap})"
            )

            def split_docs(docs: List[Document]) -> List[Document]:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                return splitter.split_documents(docs)

            texts: List[Document] = await asyncio.to_thread(split_docs, documents)

            # 3. Clean up the text chunks (replace unwanted characters)
            logger.info("Cleaning split chunks (replace tabs)")
            cleaned_texts = await self.replace_t_with_space(texts)

            # 4. Create HuggingFace embeddings and encode the cleaned text chunks into a FAISS vector store
            logger.info(
                "Initializing HuggingFaceEmbeddings and building FAISS vector store"
            )
            embeddings = HuggingFaceEmbeddings(
                model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            # FAISS.from_documents is blocking/heavy -> run in thread
            def build_faiss(docs: List[Document], emb):
                return FAISS.from_documents(docs, emb)

            vectorstore: FAISS = await asyncio.to_thread(
                build_faiss, cleaned_texts, embeddings
            )

            logger.info(f"Finished encode_book, vector store size: approx {len(texts)}")
            return vectorstore

        except Exception as e:
            logger.exception("Error in encode_book")
            raise RuntimeError("Failed to encode book into FAISS vector store") from e

    @staticmethod
    async def replace_t_with_space(list_of_documents: List[Document]) -> List[Document]:
        """
        Description:
            Replaces all tab characters ('\t') with spaces in the page content of each document.

        Params:
            list_of_documents (list[Document]): A list of Document objects with 'page_content'.

        Return:
            list[Document]: The modified list of documents with tab characters replaced by spaces.

        Exceptions:
            TypeError: If input is not a list of Document objects.
        """
        logger.info(
            f"Starting replace_t_with_space on {len(list_of_documents)} documents"
            if isinstance(list_of_documents, list)
            else 0
        )
        if not isinstance(list_of_documents, list) or not all(
            isinstance(d, Document) for d in list_of_documents
        ):
            raise TypeError("Input must be a list of Document objects")

        # This operation is light; run inline
        for doc in list_of_documents:
            if isinstance(doc.page_content, str):
                doc.page_content = doc.page_content.replace("\t", " ")
        logger.info("Finished replace_t_with_space")
        return list_of_documents

    @staticmethod
    async def encode_chapter_summaries(
        chapter_summaries_input: List[Document],
    ) -> FAISS:
        """
        Description:
            Encodes a list of chapter summaries into a FAISS vector store using HuggingFace embeddings.

        Params:
            chapter_summaries_input (list[Document]): A list of Document objects representing the chapter summaries.

        Return:
            FAISS: A FAISS vector store containing the encoded chapter summaries.

        Exceptions:
            TypeError: If chapter_summaries_input is not a list of Document objects.
            RuntimeError: If encoding fails.
        """
        logger.info(
            f"Starting encode_chapter_summaries for {len(chapter_summaries_input)} summaries"
            if isinstance(chapter_summaries_input, list)
            else 0
        )

        if not isinstance(chapter_summaries_input, list) or not all(
            isinstance(d, Document) for d in chapter_summaries_input
        ):
            raise TypeError(
                "chapter_summaries_input must be a list of Document objects"
            )

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            def build_faiss(docs: List[Document], emb):
                return FAISS.from_documents(docs, emb)

            chapter_summaries_vectorstore: FAISS = await asyncio.to_thread(
                build_faiss, chapter_summaries_input, embeddings
            )
            logger.info("Finished encode_chapter_summaries")
            return chapter_summaries_vectorstore

        except Exception as e:
            logger.exception("Error encoding chapter summaries")
            raise RuntimeError("Failed to encode chapter summaries") from e

    @staticmethod
    async def encode_quotes(book_quotes_list: List[Document]) -> Union[FAISS, List]:
        """
        Description:
            Encodes a list of book quotes into a FAISS vector store using HuggingFace embeddings.

        Params:
            book_quotes_list (list[Document]): A list of Document objects, each representing a quote from the book.

        Return:
            FAISS or list: A FAISS vector store containing the encoded book quotes if any quotes exist,
                          otherwise an empty list.

        Exceptions:
            TypeError: If book_quotes_list is not a list of Document objects.
            RuntimeError: If encoding fails.
        """
        logger.info(
            f"Starting encode_quotes, count= {len(book_quotes_list)}"
            if isinstance(book_quotes_list, list)
            else 0
        )

        if not isinstance(book_quotes_list, list) or not all(
            isinstance(d, Document) for d in book_quotes_list
        ):
            raise TypeError("book_quotes_list must be a list of Document objects")

        if len(book_quotes_list) == 0:
            logger.info("No quotes to encode; returning empty list")
            return []

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            def build_faiss(docs: List[Document], emb):
                return FAISS.from_documents(docs, emb)

            quotes_vectorstore: FAISS = await asyncio.to_thread(
                build_faiss, book_quotes_list, embeddings
            )
            logger.info("Finished encode_quotes")
            return quotes_vectorstore

        except Exception as e:
            logger.exception("Error encoding quotes")
            raise RuntimeError("Failed to encode quotes") from e

    async def create_vector_db(
        self,
        chapter_summaries_input: Tuple[Any],
        book_quotes_list: List[Document],
        hp_pdf_path: str,
    ) -> Tuple[Union[FAISS, Any], Union[FAISS, Any], Union[FAISS, List, Any]]:
        """
        Description:
            Creating Vector Stores and Retrievers for Book and Chapter Summaries.
            Loads existing FAISS stores if present, otherwise builds them and saves to disk.

        Params:
            chapter_summaries_input (list[Document]): Chapter summaries to encode.
            book_quotes_list (list[Document]): Book quotes to encode.
            hp_pdf_path (str): Path to the PDF source used for chunking/encoding.

        Return:
            tuple: (chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore)

        Exceptions:
            TypeError: If inputs are not of expected types.
            RuntimeError: If building/loading vector stores fails.
        """
        logger.info("Starting create_vector_db")

        if not isinstance(chapter_summaries_input, list) or not all(
            isinstance(d, Document) for d in chapter_summaries_input
        ):
            raise TypeError(
                "chapter_summaries_input must be a list of Document objects"
            )
        if not isinstance(book_quotes_list, list) or not all(
            isinstance(d, Document) for d in book_quotes_list
        ):
            raise TypeError("book_quotes_list must be a list of Document objects")
        if not isinstance(hp_pdf_path, str):
            raise TypeError("hp_pdf_path must be a string")

        chunks_vector_store: Optional[FAISS] = None
        chapter_summaries_vector_store: Optional[FAISS] = None
        book_quotes_vectorstore: Union[FAISS, List, Any] = []

        try:
            # If vector stores exist on disk, load them
            if (
                os.path.exists("chunks_vector_store")
                and os.path.exists("chapter_summaries_vector_store")
                and os.path.exists("book_quotes_vectorstore")
            ):
                logger.info("Found existing vector stores; loading from disk")
                embeddings = HuggingFaceEmbeddings(
                    model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                )

                def load_faiss(folder: str, emb):
                    return FAISS.load_local(
                        folder, emb, allow_dangerous_deserialization=True
                    )

                chunks_vector_store = await asyncio.to_thread(
                    load_faiss, "chunks_vector_store", embeddings
                )
                chapter_summaries_vector_store = await asyncio.to_thread(
                    load_faiss, "chapter_summaries_vector_store", embeddings
                )
                book_quotes_vectorstore = await asyncio.to_thread(
                    load_faiss, "book_quotes_vectorstore", embeddings
                )

                logger.info("Loaded vector stores from disk successfully")
            else:
                logger.info("Vector stores not found; encoding fresh vector stores")

                # 1. Encode the book into a vector store of chunks
                chunks_vector_store = await self.encode_book(
                    hp_pdf_path, chunk_size=1000, chunk_overlap=200
                )

                # 2. Encode the chapter summaries into a vector store
                chapter_summaries_vector_store = await self.encode_chapter_summaries(
                    chapter_summaries_input
                )

                # 3. Encode the book quotes into a vector store
                book_quotes_vectorstore = await self.encode_quotes(book_quotes_list)

                # 4. Save the vector stores to disk for future use (blocking -> to_thread)
                logger.info("Saving vector stores to disk")
                if isinstance(chunks_vector_store, FAISS):
                    await asyncio.to_thread(
                        chunks_vector_store.save_local, "chunks_vector_store"
                    )
                if isinstance(chapter_summaries_vector_store, FAISS):
                    await asyncio.to_thread(
                        chapter_summaries_vector_store.save_local,
                        "chapter_summaries_vector_store",
                    )
                if isinstance(book_quotes_vectorstore, FAISS):
                    await asyncio.to_thread(
                        book_quotes_vectorstore.save_local, "book_quotes_vectorstore"
                    )

                logger.info("Saved vector stores to disk successfully")

        except Exception as e:
            logger.exception("Error in create_vector_db")
            raise RuntimeError("Failed to create or load vector DBs") from e

        logger.info("Finished create_vector_db")
        return (
            chunks_vector_store,
            chapter_summaries_vector_store,
            book_quotes_vectorstore,
        )


# if __name__ == "__main__":
#     handler = EncodeEmbeddings()
#     hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
#     A, B, C = handler.create_vector_db(chapter_summaries, [], hp_pdf_path)
#     print((A, B, C))
#     print("done")
