from time import monotonic
import PyPDF2
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
        # Define the path to the Harry Potter PDF file.
        self.input_pdf_path = input_pdf_path

    def split_into_chapters(self):
        """
        Splits a PDF book into chapters based on chapter title patterns.
        # This function takes the path to the PDF and returns a list of Document objects,
        each representing a chapter.

        Returns:
            list: A list of Document objects, each representing a chapter with its
                  text content and chapter number metadata.
        """
        with open(self.input_pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            documents = pdf_reader.pages

            # Concatenate text from all pages
            text = " ".join([doc.extract_text() for doc in documents])

            # Split text into chapters based on chapter title pattern
            chapters = re.split(r'(CHAPTER\s[A-Z]+(?:\s[A-Z]+)*)', text)

            # Create Document objects with chapter metadata
            chapter_docs = []
            chapter_num = 1
            for i in range(1, len(chapters), 2):
                chapter_text = chapters[i] + chapters[i + 1]  # Combine title and content
                doc = Document(page_content=chapter_text, metadata={"chapter": chapter_num})
                chapter_docs.append(doc)
                chapter_num += 1

        return chapter_docs

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
    def extract_book_quotes_as_documents(documents, min_length=50):
        """
        Extracts quotes from documents and returns them as separate Document objects.

        Args:
            documents (list): List of Document objects to extract quotes from.
            min_length (int, optional): Minimum length of quotes to extract. Defaults to 50.

        Returns:
            list: List of Document objects containing extracted quotes.
        """
        quotes_as_documents = []
        # Pattern for quotes longer than min_length characters, including line breaks
        quote_pattern_longer_than_min_length = re.compile(rf'"(.{{{min_length},}}?)"', re.DOTALL)

        for doc in documents:
            content = doc.page_content
            content = content.replace('\n', ' ')
            found_quotes = quote_pattern_longer_than_min_length.findall(content)

            for quote in found_quotes:
                quote_doc = Document(page_content=quote)
                quotes_as_documents.append(quote_doc)

        return quotes_as_documents

    @staticmethod
    def replace_double_lines_with_one_line(text):
        """
        Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

        Args:
            text (str): The input text string.

        Returns:
            str: The text string with double newlines replaced by single newlines.
        """
        cleaned_text = re.sub(r'\n\n', '\n', text)
        return cleaned_text

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """
        Calculates the number of tokens in a given string using a specified encoding.

        Args:
            string (str): The input string to tokenize.
            encoding_name (str): The name of the encoding to use (e.g., 'cl100k_base').

        Returns:
            int: The number of tokens in the string according to the specified encoding.
        """
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def create_chapter_summary(self, chapter):
        """
        Creates a summary of a chapter using a large language model (LLM).
            Args:
                chapter: A Document object representing the chapter to summarize.

            Returns:
                A Document object containing the summary of the chapter.
        """

        """ Summarization Prompt Template for LLM-based Summarization"""

        summarization_prompt_template = """
            Write an extensive summary of the following:

            {text}

            SUMMARY:
            """
        # Create a PromptTemplate object using the template string.
        # The input variable "text" will be replaced with the content to summarize.
        summarization_prompt = PromptTemplate(
            template=summarization_prompt_template,
            input_variables=["text"]
        )
        # Extract the text content from the chapter
        chapter_txt = chapter.page_content

        # Specify the LLM model and configuration
        llm = AzureOpenAIModels().get_azure_model_4()
        gpt_4o_mini_max_tokens = 50000  # Maximum token limit for the model
        verbose = False  # Set to True for more detailed output
        model_name = "gpt-35-turbo-"

        # Calculate the number of tokens in the chapter text
        num_tokens = self.num_tokens_from_string(chapter_txt, encoding_name=model_name)

        # Choose the summarization chain type based on token count
        if num_tokens < gpt_4o_mini_max_tokens:
            # For shorter chapters, use the "stuff" chain type
            chain = load_summarize_chain(
                llm,
                chain_type="stuff",
                prompt=summarization_prompt,
                verbose=verbose
            )
        else:
            # For longer chapters, use the "map_reduce" chain type
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=summarization_prompt,
                combine_prompt=summarization_prompt,
                verbose=verbose
            )

        # Start timer to measure summarization time
        start_time = monotonic()

        # Create a Document object for the chapter
        doc_chapter = Document(page_content=chapter_txt)

        # Generate the summary using the selected chain
        summary_result = chain.invoke({"input_documents": [doc_chapter]})

        # Print chain type and execution time for reference
        logger.info(f"Chain type: {chain.__class__.__name__}")
        logger.info(f"Run time: {monotonic() - start_time}")

        # Clean up the summary text (remove double newlines, etc.)
        summary_text = self.replace_double_lines_with_one_line(summary_result["output_text"])

        # Create a Document object for the summary, preserving chapter metadata
        doc_summary = Document(page_content=summary_text, metadata=chapter.metadata)

        return doc_summary

    def preprocess_pipeline(self):
        """Splitting the PDF into Chapters and Preprocessing"""
        # 1. Split the PDF into chapters using the provided helper function.

        chapters = self.split_into_chapters()

        # 2. Clean up the text in each chapter by replacing unwanted characters (e.g., '\t') with spaces.
        #    This ensures the text is consistent and easier to process downstream.
        chapters = self.replace_t_with_space(chapters)

        # 3. Print the number of chapters extracted to verify the result.
        logger.info(f"length of Chapters are: {len(chapters)}")

        """--- Load and Preprocess the PDF, then Extract Quotes ---"""

        # 1. Load the PDF using PyPDFLoader
        loader = PyPDFLoader(self.input_pdf_path)
        document = loader.load()

        # 2. Clean the loaded document by replacing unwanted characters (e.g., '\t') with spaces
        document_cleaned = self.replace_t_with_space(document)

        # 3. Extract a list of quotes from the cleaned document as Document objects
        book_quotes_list = self.extract_book_quotes_as_documents(document_cleaned)

        logger.info(f"Book Quotes List: {book_quotes_list}")

        """--- Generate Summaries for Each Chapter ---"""

        # Initialize an empty list to store the summaries of each chapter
        chapter_summaries = []

        # Iterate over each chapter in the chapters list
        for chapter in chapters:
            # Generate a summary for the current chapter using the create_chapter_summary function
            summary = self.create_chapter_summary(chapter)
            # Append the summary to the chapter_summaries list
            chapter_summaries.append(summary)

        return chapter_summaries, book_quotes_list


if __name__ == "__main__":

    hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
    handler = ProcessDocument(hp_pdf_path)
    summaries, book_quotes_list = handler.preprocess_pipeline()
    for index, value in enumerate(summaries):
        print((value.metadata, value.page_content))
