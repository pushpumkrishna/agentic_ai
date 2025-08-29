import asyncio
from time import monotonic
from typing import Any
import pdfplumber
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
import regex as re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import tiktoken
from backend.config.azure_models import AzureOpenAIModels
from backend.config.logging_lib import logger


class ProcessDocument:
    def __init__(self, input_pdf_path: str):
        """
        Description:
            Initialize the ProcessDocument handler with the PDF path.

        Params:
            input_pdf_path (str): Path to the input PDF file.

        Return:
            None

        Exceptions:
            TypeError: If input_pdf_path is not a string.
        """
        if not isinstance(input_pdf_path, str):
            raise TypeError("input_pdf_path must be a string")
        self.input_pdf_path = input_pdf_path
        logger.info(
            f"Initialized ProcessDocument with file path: {self.input_pdf_path}"
        )

    async def split_into_chapters(self) -> list[Document]:
        """
        Description:
            Splits a PDF into chapters based on regex chapter patterns using pdfplumber.

        Params:
            None

        Return:
            list[Document]: List of Document objects with chapter text and metadata.

        Exceptions:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If no text can be extracted from the PDF.
        """
        logger.info(f"Starting chapter split for PDF: {self.input_pdf_path}")

        try:
            async with asyncio.Semaphore(1):  # Prevent race condition on file read
                # Use asyncio.to_thread for blocking pdfplumber operations
                def extract_text_pdfplumber(path):
                    full_text = ""
                    with pdfplumber.open(path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text() or ""
                                full_text += page_text + " "
                            except Exception as e:
                                logger.warning(
                                    f"Failed to extract text from page {i}: {e}"
                                )
                    return full_text

                text = await asyncio.to_thread(
                    extract_text_pdfplumber, self.input_pdf_path
                )

        except FileNotFoundError:
            logger.error(f"File not found: {self.input_pdf_path}")
            raise
        except Exception as e:
            logger.exception("Unexpected error while splitting chapters")
            raise

        if not text.strip():
            raise ValueError("No text could be extracted from PDF.")

        # Split into chapters using regex
        chapters = await asyncio.to_thread(
            re.split, r"(CHAPTER\s[A-Z]+(?:\s[A-Z]+)*)", text
        )
        chapter_docs = []
        chapter_num = 1
        for i in range(1, len(chapters), 2):
            chapter_text = chapters[i] + chapters[i + 1]
            doc = Document(page_content=chapter_text, metadata={"chapter": chapter_num})
            chapter_docs.append(doc)
            chapter_num += 1

        logger.info(f"Finished splitting into {len(chapter_docs)} chapters")
        return chapter_docs

    @staticmethod
    async def replace_t_with_space(list_of_documents: list[Document]) -> list[Document]:
        """
        Description:
            Replace all tabs with spaces in document text.

        Params:
            list_of_documents (list[Document]): Documents to clean.

        Return:
            list[Document]: Cleaned documents.

        Exceptions:
            TypeError: If input is not a list of Document objects.
        """
        logger.info("Starting tab replacement in documents")
        if not isinstance(list_of_documents, list) or not all(
            isinstance(d, Document) for d in list_of_documents
        ):
            raise TypeError("Input must be a list of Document objects")
        for doc in list_of_documents:
            doc.page_content = doc.page_content.replace("\t", " ")
        logger.info("Finished tab replacement in documents")
        return list_of_documents

    @staticmethod
    async def extract_book_quotes_as_documents(
        documents: list[Document], min_length: int = 50
    ) -> list[Document]:
        """
        Description:
            Extract long quotes from documents.

        Params:
            documents (list[Document]): Input documents.
            min_length (int): Minimum length of a quote.

        Return:
            list[Document]: Extracted quotes as documents.

        Exceptions:
            TypeError: If documents is not a list of Document objects.
        """
        logger.info("Starting quote extraction")
        if not isinstance(documents, list) or not all(
            isinstance(d, Document) for d in documents
        ):
            raise TypeError("documents must be a list of Document objects")
        quotes_as_documents = []
        quote_pattern = re.compile(rf'"(.{{{min_length},}}?)"', re.DOTALL)

        for doc in documents:
            content = doc.page_content.replace("\n", " ")
            found_quotes = await asyncio.to_thread(quote_pattern.findall, content)
            for quote in found_quotes:
                quotes_as_documents.append(Document(page_content=quote))

        logger.info("Extracted %d quotes: {len(quotes_as_documents)}")
        return quotes_as_documents

    @staticmethod
    async def replace_double_lines_with_one_line(text: str) -> str:
        """
        Description:
            Replace double newlines with single newlines.

        Params:
            text (str): Input text.

        Return:
            str: Cleaned text.

        Exceptions:
            TypeError: If input is not string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cleaned_text = await asyncio.to_thread(re.sub, r"\n\n", "\n", text)
        return cleaned_text

    @staticmethod
    async def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """
        Description:
            Count tokens in a string with tiktoken.

        Params:
            string (str): Text to encode.
            encoding_name (str): Encoding model name.

        Return:
            int: Number of tokens.

        Exceptions:
            TypeError: If inputs are not strings.
        """
        if not isinstance(string, str) or not isinstance(encoding_name, str):
            raise TypeError("Both string and encoding_name must be strings")
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = await asyncio.to_thread(lambda: len(encoding.encode(string)))
        return num_tokens

    async def create_chapter_summary(self, chapter: Document) -> Document:
        """
        Description:
            Generate a summary for a chapter.

        Params:
            chapter (Document): Chapter document.

        Return:
            Document: Summary document.

        Exceptions:
            TypeError: If chapter is not a Document object.
            RuntimeError: If summarization chain fails.
        """
        logger.info(f"Starting chapter summarization for chapter : {chapter.metadata}")
        if not isinstance(chapter, Document):
            raise TypeError("chapter must be a Document instance")
        summarization_prompt_template = """
            Write an extensive summary of the following:

            {text}

            SUMMARY:
        """
        summarization_prompt = PromptTemplate(
            template=summarization_prompt_template, input_variables=["text"]
        )
        chapter_txt = chapter.page_content

        llm = AzureOpenAIModels().get_azure_model_4()
        gpt_4o_mini_max_tokens = 50000
        model_name = "gpt-35-turbo-"
        num_tokens = await self.num_tokens_from_string(
            chapter_txt, encoding_name=model_name
        )

        if num_tokens < gpt_4o_mini_max_tokens:
            chain = load_summarize_chain(
                llm, chain_type="stuff", prompt=summarization_prompt, verbose=False
            )
        else:
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=summarization_prompt,
                combine_prompt=summarization_prompt,
                verbose=False,
            )
        start_time = monotonic()
        doc_chapter = Document(page_content=chapter_txt)

        try:
            summary_result = await asyncio.to_thread(
                chain.invoke, {"input_documents": [doc_chapter]}
            )
        except Exception as e:
            logger.exception("Error during summarization")
            raise RuntimeError("Summarization failed") from e

        logger.info(f"Chain type: {chain.__class__.__name__}")
        logger.info(f"Run time: {monotonic() - start_time}")
        summary_text = await self.replace_double_lines_with_one_line(
            summary_result["output_text"]
        )
        doc_summary = Document(page_content=summary_text, metadata=chapter.metadata)
        logger.info(f"Finished summarization for chapter: {chapter.metadata}")
        return doc_summary

    async def preprocess_pipeline(self) -> tuple[tuple[Any], list[Document]]:
        """
        Description:
            Full preprocessing pipeline: split, clean, extract quotes, summarize.

        Params:
            None

        Return:
            tuple[list[Document], list[Document]]: Chapter summaries and extracted quotes.

        Exceptions:
            RuntimeError: If any processing step fails.
        """
        logger.info("Starting preprocessing pipeline")
        try:
            chapters = await self.split_into_chapters()
            chapters = await self.replace_t_with_space(chapters)
            logger.info(f"length of Chapters are: {len(chapters)}")

            loader = PyPDFLoader(self.input_pdf_path)
            document = await asyncio.to_thread(loader.load)
            document_cleaned = await self.replace_t_with_space(document)
            book_quotes_list = await self.extract_book_quotes_as_documents(
                document_cleaned
            )
            logger.info(f"Book Quotes List length: {len(book_quotes_list)}")

            chapter_summaries = await asyncio.gather(
                *[self.create_chapter_summary(ch) for ch in chapters]
            )
        except Exception as e:
            logger.exception("Error in preprocessing pipeline")
            raise RuntimeError("Pipeline failed") from e

        logger.info("Finished preprocessing pipeline")
        return chapter_summaries, book_quotes_list


# if __name__ == "__main__":
#     hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
#     handler = ProcessDocument(hp_pdf_path)
#     # summaries, book_quotes_list = await handler.preprocess_pipeline()
#     for index, value in enumerate(summaries):
#         print((value.metadata, value.page_content))
