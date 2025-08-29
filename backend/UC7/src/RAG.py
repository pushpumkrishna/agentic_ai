import uuid
from operator import itemgetter
import bs4
import requests
from langchain.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.stores import InMemoryByteStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
import warnings
from typing import List, Dict
from langchain.load import dumps, loads
from pydantic import BaseModel
from torch import cosine_similarity
import torch
from pydantic import Field, BaseModel
from ragatouille import RAGPretrainedModel

from backend.config.azure_models import AzureOpenAIModels
from backend.config.logging_lib import logger

warnings.filterwarnings("ignore")


class RAGChainBuilder:
    def __init__(
        self,
        query: str = " ",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_dir: str = "./chroma_db",
    ):
        """
        Initializes the RAGChainBuilder with model and chunking parameters.

        Args:
            chunk_size (int): The size of text chunks.
            chunk_overlap (int): The overlap between chunks.
            persist_dir (str): The directory to persist the Chroma vector store.
        """
        self.llm = AzureOpenAIModels().get_azure_model_4()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
            },
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.persist_dir = persist_dir
        self.document = self._load_documents()
        self.query = query
        self.retriever = self._get_local_retriever()

    @staticmethod
    def _load_documents() -> List:
        """Loads and parses web content from a URL."""
        logger.info("Loading and parsing web content...")
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )

        document = loader.load()
        return document

    @staticmethod
    def format_docs(docs) -> str:
        """A static helper method to format documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_local_retriever(self):
        logger.info("Splitting documents into chunks...")
        splits = self.text_splitter.split_documents(self.document)

        logger.info("Creating embedding model and vector store...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
        )
        return vectorstore.as_retriever()

    def process_and_create_retriever_v1(self):
        """Splits documents and creates a Chroma retriever."""

        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {
                # only take the "question" field and send it to retriever
                "context": itemgetter("question")
                | self.retriever
                | RunnableLambda(self.format_docs),
                "question": itemgetter("question"),
            }
            # {"context": self.retriever | RunnableLambda(self.format_docs), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        final_response = rag_chain.invoke({"question": self.query})
        logger.info(f"process_and_create_retriever_v1 response: {final_response}")

    def process_and_create_retriever_v2_generate_questions(self):
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, chunk_overlap=50
        )
        splits = text_splitter.split_documents(self.document)

        # Index the chunks in a Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.embedding_model
        )

        retriever = vectorstore.as_retriever()

        # Prompt for generating multiple queries
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"question": RunnablePassthrough()}
            | prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        final_response = rag_chain.invoke({"question": self.query})
        logger.info(
            f"process_and_create_retriever_v2_generate_questions response: {final_response}"
        )

        """
        The LLM has rephrased our original question using different keywords like “break down complex tasks”, 
        “methods”, and “process.” We can retrieve documents for all of these queries and combine the results. 
        A simple way to combine them is to take the unique set of all retrieved documents.
        """

        def get_unique_union(documents: list[list]):
            """A simple function to get the unique union of retrieved documents"""
            # Flatten the list of lists and convert each Document to a string for uniqueness
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            unique_docs = list(set(flattened_docs))
            return [loads(doc) for doc in unique_docs]

        # Build the retrieval chain
        retrieval_chain = rag_chain | Runnable.map(retriever) | get_unique_union

        # Invoke the chain and check the number of documents retrieved
        docs = retrieval_chain.invoke({"question": self.query})
        logger.info(f"Total unique documents retrieved: {len(docs)}")

        # The final RAG chain
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = (
            {"context": retrieval_chain, "question": itemgetter("question")}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        final_rag_chain.invoke({"question": self.query})
        logger.info(
            f"process_and_create_retriever_v2_generate_questions final response: {final_response}"
        )

    @staticmethod
    def reciprocal_rank_fusion_re_ranking(results: list[list], k=60) -> List:
        """Reciprocal Rank Fusion that intelligently combines multiple ranked lists"""
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # The core of RRF: documents ranked higher (lower rank value) get a larger score
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort documents by their new fused scores in descending order
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return reranked_results

    def process_reciprocal_rank_fusion(self) -> None:
        # Use a slightly different prompt for RAG-Fusion
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion | self.llm | StrOutputParser() | (lambda x: x.split("\n"))
        )

        # Build the new retrieval chain with RRF
        # noinspection PyTypeChecker
        retrieval_chain_rag_fusion = (
            generate_queries
            | Runnable.map(self.retriever)
            | self.reciprocal_rank_fusion_re_ranking
        )
        docs = retrieval_chain_rag_fusion.invoke({"question": self.query})
        logger.info(f"Total re-ranked documents retrieved: {len(docs)}")

    def process_decomposition(self):
        """
        The Decomposition technique uses an LLM to break down a complex query into a set of simpler,
        self-contained sub-questions. Then answer each one and synthesize a final answer.

        :return:
        """
        # Decomposition prompt
        template = """
        You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):
        """
        prompt_decomposition = ChatPromptTemplate.from_template(template)

        # Chain to generate sub-questions
        generate_queries_decomposition = (
            prompt_decomposition
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        # Generate and print the sub-questions
        question = (
            "What are the main components of an LLM-powered autonomous agent system?"
        )
        sub_questions = generate_queries_decomposition.invoke({"question": question})
        print(sub_questions)

        # RAG prompt
        prompt_rag = hub.pull("rlm/rag-prompt")

        # A list to hold the answers to our sub-questions
        rag_results = []
        for sub_question in sub_questions:
            # Retrieve documents for each sub-question
            retrieved_docs = self.retriever.get_relevant_documents(sub_question)

            # Use our standard RAG chain to answer the sub-question
            answer = (prompt_rag | self.llm | StrOutputParser()).invoke(
                {"context": retrieved_docs, "question": sub_question}
            )
            rag_results.append(answer)

        def format_qa_pairs(questions, answers):
            """Format Q and A pairs"""
            formatted_string = ""
            for i, (one_question, one_answer) in enumerate(
                zip(questions, answers), start=1
            ):
                formatted_string += (
                    f"Question {i}: {one_question}\nAnswer {i}: {one_answer}\n\n"
                )
            return formatted_string.strip()

        # Format the Q&A pairs into a single context string
        context = format_qa_pairs(sub_questions, rag_results)

        # Final synthesis prompt
        template = """Here is a set of Q+A pairs:

        {context}

        Use these to synthesize an answer to the original question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = prompt | self.llm | StrOutputParser()

        final_rag_chain.invoke({"context": context, "question": question})

    def process_step_back_prompting(self):
        """
        Sometimes, a user’s query is too specific, while our documents contain the more general, underlying information
        needed to answer it.
        A direct search for this might fail. The Step-Back technique uses an LLM to take a “step back” and form a
        more general question, like “What are the powers and duties of the band The Police?”
        We then retrieve context for both the specific and general questions, providing a richer context for the
        final answer
        :return:
        """

        # Few-shot examples to teach the model how to generate step-back (more generic) questions
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel's was born in what country?",
                "output": "what is Jan Sindel's personal history?",
            },
        ]

        # Define how each example is formatted in the prompt
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),  # User input
                ("ai", "{output}"),  # Model's response
            ]
        )

        # Wrap the few-shot examples into a reusable prompt template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        # Full prompt includes system instruction, few-shot examples, and the user question
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at world knowledge. Your task is to step back and paraphrase a question "
                    "to a more generic step-back question, which is easier to answer. Here are a few examples:",
                ),
                few_shot_prompt,
                ("user", "{question}"),
            ]
        )

        # Define a chain to generate step-back questions using the prompt and an OpenAI model
        generate_queries_step_back = prompt | self.llm | StrOutputParser()

        # Run the chain on a specific question
        step_back_question = generate_queries_step_back.invoke({"question": self.query})

        # Output the original and generated step-back question
        print(f"Original Question: {self.query}")
        print(f"Step-Back Question: {step_back_question}")

        from langchain_core.runnables import RunnableLambda

        # Prompt for the final response
        response_prompt_template = """
        You are an expert of world knowledge. I am going to ask you a question. 
        Your response should be comprehensive and not contradicted with the following context if they are relevant. 
        Otherwise, ignore them if they are not relevant.

        # Normal Context
        {normal_context}

        # Step-Back Context
        {step_back_context}

        # Original Question: {question}
        # Answer:
        
        """
        response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

        # The full chain
        chain = (
            {
                # Retrieve context using the normal question
                "normal_context": RunnableLambda(lambda x: x["question"])
                | self.retriever,
                # Retrieve context using the step-back question
                "step_back_context": generate_queries_step_back | self.retriever,
                # Pass on the original question
                "question": lambda x: x["question"],
            }
            | response_prompt
            | self.llm
            | StrOutputParser()
        )

        chain.invoke({"question": self.query})

    def process_hypothetical_document_embeddings(self):
        # HyDE prompt
        template = """Please write a scientific paper passage to answer the question
        Question: {question}
        Passage:"""
        prompt_hyde = ChatPromptTemplate.from_template(template)

        # Chain to generate the hypothetical document
        generate_docs_for_retrieval = prompt_hyde | self.llm | StrOutputParser()

        # Generate and print the hypothetical document
        hypothetical_document = generate_docs_for_retrieval.invoke(
            {"question": self.query}
        )
        print(hypothetical_document)

        """
        This passage is a perfect, textbook-style answer. Now, we use its embedding to find real documents.
        """

        # Retrieve documents using the HyDE approach
        retrieval_chain = generate_docs_for_retrieval | self.retriever
        retrieved_docs = retrieval_chain.invoke({"question": self.query})

        final_rag_chain = (
            {"context": retrieval_chain, "question": itemgetter("question")}
            | prompt_hyde
            | self.llm
            | StrOutputParser()
        )

        # Use our standard RAG chain to generate the final answer from the retrieved context
        response = final_rag_chain.invoke(
            {"context": retrieved_docs, "question": self.query}
        )
        print(response)

    def process_routing_n_query_construction(self):
        """
        Logical routing works perfectly when you have clearly defined categories.
        But what if you want to route based on the style or domain of a question? For example, you might want to answer physics questions with a serious, academic tone and math questions with a step-by-step, pedagogical approach. This is where Semantic Routing comes in

        :return:
        """

        # A prompt for a physics expert
        physics_template = """You are a very smart physics professor. \
        You are great at answering questions about physics in a concise and easy to understand manner. \
        When you don't know the answer to a question you admit that you don't know.

        Here is a question:
        {query}"""

        # A prompt for a math expert
        math_template = """You are a very good mathematician. You are great at answering math questions. \
        You are so good because you are able to break down hard problems into their component parts, \
        answer the component parts, and then put them together to answer the broader question.

        Here is a question:
        {query}"""

        # Store our templates and their embeddings for comparison
        prompt_templates = [physics_template, math_template]
        prompt_embeddings = self.embedding_model.embed_documents(prompt_templates)

        def prompt_router(input: Dict):
            """A function to route the input query to the most similar prompt template."""
            # 1. Embed the incoming user query
            query_embedding = self.embedding_model.embed_query(input["query"])

            # Convert the embeddings to tensors
            query_tensor = torch.tensor(query_embedding)
            prompt_tensor = torch.tensor(prompt_embeddings)

            # 2. Compute the cosine similarity between the query and all prompt templates
            similarity = cosine_similarity(query_tensor, prompt_tensor)[0]

            # 3. Find the index of the most similar prompt
            most_similar_index = similarity.argmax()

            # 4. Select the most similar prompt template
            chosen_prompt = prompt_templates[most_similar_index]

            print(
                f"DEBUG: Using {'MATH' if most_similar_index == 1 else 'PHYSICS'} template."
            )

            # 5. Return the chosen prompt object
            return PromptTemplate.from_template(chosen_prompt)

        # The final chain that combines the router with the LLM
        chain = (
            {"query": RunnablePassthrough()}
            | RunnableLambda(prompt_router)  # Dynamically select the prompt
            | self.llm
            | StrOutputParser()
        )

        # Ask a physics question
        print(chain.invoke({"question": self.query}))

    def process_query_structuring(self):
        """
        Query Structuring is the technique of converting a natural language question into a structured query
        that can use these metadata filters for highly precise retrieval
        :return:
        """

        # Load a YouTube transcript to inspect its metadata
        docs = YoutubeLoader.from_youtube_url(
            "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
        ).load()

        # Print the metadata of the first document
        print(docs[0].metadata)

        """
        This document has rich metadata: view_count, publish_date, length. We want our users to be able to 
        filter on these fields using natural language. 
        To do this, we'll define another Pydantic schema, this time for a structured video search query
        """

    def process_advanced_indexing_strategies(self):
        """
        The core idea of Multi-Representation Indexing is simple but powerful:
        instead of embedding the full document chunks, we create a smaller, more focused representation of each chunk
        (like a summary) and embed that instead)

        During retrieval, we search over these concise summaries.
        Once we find the best summary, we use its ID to look up and retrieve the full, original document chunk.
        :return:
        """

        # The chain for generating summaries
        summary_chain = (
            # Extract the page_content from the document object
            {"doc": lambda x: x.page_content}
            # Pipe it into a prompt template
            | ChatPromptTemplate.from_template(
                "Summarize the following document:\n\n{doc}"
            )
            # Use an LLM to generate the summary
            | self.llm
            # Parse the output into a string
            | StrOutputParser()
        )

        # Use .batch() to run the summarization in parallel for efficiency
        summaries = summary_chain.batch(self.document, {"max_concurrency": 5})

        # Let's inspect the first summary
        print(summaries[0])

        """
        We need a MultiVectorRetriever which requires two main components:

            1. A vectorstore to store the embeddings of our summaries.
            2. A docstore (a simple key-value store) to hold the original, full documents.
        """

        # The vectorstore to index the summary embeddings
        vectorstore = Chroma(
            collection_name="summaries", embedding_function=self.embedding_model
        )

        # The storage layer for the parent documents
        store = InMemoryByteStore()
        id_key = "doc_id"  # This key will link summaries to their parent documents

        # The retriever that orchestrates the whole process
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
        )

        # Generate unique IDs for each of our original documents
        doc_ids = [str(uuid.uuid4()) for _ in self.document]

        # Create new Document objects for the summaries, adding the 'doc_id' to their metadata
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]

        # Add the summaries to the vectorstore
        retriever.vectorstore.add_documents(summary_docs)

        # Add the original documents to the docstore, linking them by the same IDs
        retriever.docstore.mset(list(zip(doc_ids, self.document)))

        query = "Memory in agents"

        # First, let's see what the vectorstore finds by searching the summaries
        sub_docs = vectorstore.similarity_search(query, k=1)
        print("--- Result from searching summaries ---")
        print(sub_docs[0].page_content)
        print("\n--- Metadata showing the link to the parent document ---")
        print(sub_docs[0].metadata)

        # Let the full retriever do its job
        retrieved_docs = retriever.get_relevant_documents(query, n_results=1)

        # Print the beginning of the retrieved full document
        print("\n--- The full document retrieved by the MultiVectorRetriever ---")
        print(retrieved_docs[0].page_content[0:500])

    def process_hierarchical_indexing_raptor(self):
        """
        RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) takes the multi-representation idea a
        step further. Instead of just one layer of summaries, RAPTOR builds a multi-level tree of summaries.
        It starts by clustering small document chunks. It then summarizes each cluster.
        :return:
        """
        pass

    def process_token_level_precision_colbert(self):
        """
        Standard embedding models create a single vector for an entire chunk of text
        (this is called a “bag-of-words” approach)

        ColBERT (Contextualized Late Interaction over BERT) offers a more granular approach.
        It generates a separate, context-aware embedding for every single token in the document

        When you make a query, ColBERT also embeds every token in your query.
        Then, instead of comparing one document vector to one query vector,
        it finds the maximum similarity between each query token and any document token.

        This “late interaction” allows for a much finer-grained understanding of relevance, excelling at
        keyword-style searches.
        :return:
        """

        # Load a pre-trained ColBERT model
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

        def get_wikipedia_page(title: str):
            """A helper function to retrieve content from Wikipedia."""
            # Wikipedia API endpoint and parameters
            URL = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            }
            headers = {"User-Agent": "MyRAGApp/1.0"}
            response = requests.get(URL, params=params, headers=headers)
            data = response.json()
            page = next(iter(data["query"]["pages"].values()))
            return page.get("extract")

        full_document = get_wikipedia_page("Hayao_Miyazaki")

        # Index the document with RAGatouille. It handles the chunking and token-level embedding internally.
        RAG.index(
            collection=[full_document],
            index_name="Miyazaki-ColBERT",
            max_document_length=180,
            split_documents=True,
        )

        # Search the ColBERT index
        results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
        print(results)

        # Convert the RAGatouille model into a LangChain-compatible retriever
        colbert_retriever = RAG.as_langchain_retriever(k=3)

        # Use it like any other retriever
        retrieved_docs = colbert_retriever.invoke(
            "What animation studio did Miyazaki found?"
        )
        print(retrieved_docs[0].page_content)

    def process_dedicated_re_ranking(self):
        # Load, split, and index the document
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",)
        )
        blog_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, chunk_overlap=50
        )
        splits = text_splitter.split_documents(blog_docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.embedding_model
        )

        # First-pass retriever: get the top 10 potentially relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # You will need to install cohere: pip install cohere
        # And set your COHERE_API_KEY environment variable
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CohereRerank

        # Initialize the Cohere Rerank model
        compressor = CohereRerank()

        # Create the compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        # Let's test it with our query
        question = "What is task decomposition for LLM agents?"
        compressed_docs = compression_retriever.get_relevant_documents(question)

        # Print the re-ranked documents
        print("--- Re-ranked and Compressed Documents ---")
        for doc in compressed_docs:
            print(f"Relevance Score: {doc.metadata['relevance_score']:.4f}")
            print(f"Content: {doc.page_content[:150]}...\n")

    def process_self_correction_using_ai_agents(self):
        """
        idea behind self-correcting RAG architectures like CRAG (Corrective RAG) and Self-RAG

        CRAG: If the retrieved documents are irrelevant or ambiguous for a given query, a CRAG system won’t just
        pass them to the LLM. Instead, it triggers a new, more robust web search to find better information,
        corrects the retrieved documents, and then proceeds with generation.

        Self-RAG: This approach takes it a step further. At each step, it uses an LLM to generate “reflection tokens”
        that critique the process. It grades the retrieved documents for relevance. If they’re not relevant,
        it retrieves again. Once it has good documents, it generates an answer and then grades that answer for
        factual consistency, ensuring it’s grounded in the source documents.
        :return:
        """

        # We'll use a powerful LLM like gpt-4o to act as our "judge" for reliable evaluation.
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

        # Define the output schema for our evaluation score to ensure consistent, structured output.
        class ResultScore(BaseModel):
            score: float = Field(
                ...,
                description="The score of the result, ranging from 0 to 1 where 1 is the best possible score.",
            )

        # This prompt template clearly instructs the LLM on how to score the answer's correctness.
        correctness_prompt = PromptTemplate(
            input_variables=["question", "ground_truth", "generated_answer"],
            template="""
            Question: {question}
            Ground Truth: {ground_truth}
            Generated Answer: {generated_answer}

            Evaluate the correctness of the generated answer compared to the ground truth.
            Score from 0 to 1, where 1 is perfectly correct and 0 is completely incorrect.

            Score:
            """,
        )

        # We build the evaluation chain by piping the prompt to the LLM with structured output.
        correctness_chain = correctness_prompt | llm.with_structured_output(ResultScore)

        def evaluate_correctness(_question, _ground_truth, _generated_answer):
            """A helper function to run our custom correctness evaluation chain."""
            result = correctness_chain.invoke(
                {
                    "question": _question,
                    "ground_truth": _ground_truth,
                    "generated_answer": _generated_answer,
                }
            )
            return result.score

        # Test the correctness chain with a partially correct answer.
        question = "What is the capital of France and Spain?"
        ground_truth = "Paris and Madrid"
        generated_answer = "Paris"
        score = evaluate_correctness(question, ground_truth, generated_answer)

        print(f"Correctness Score: {score}")

    # Define the output schema for our evaluation score to ensure consistent, structured output.

    def run_pipeline(self):
        # logger.info("Starting process_and_create_retriever_v1")
        # self.process_and_create_retriever_v1()
        # logger.info("Finished process_and_create_retriever_v1")

        # logger.info("Starting process_and_create_retriever_v2_generate_questions")
        # self.process_and_create_retriever_v2_generate_questions()
        # logger.info("Finished process_and_create_retriever_v2_generate_questions")
        #
        # logger.info("Starting process_reciprocal_rank_fusion")
        # self.process_reciprocal_rank_fusion()
        # logger.info("Finished process_reciprocal_rank_fusion")
        #
        # logger.info("Starting process_decomposition")
        # self.process_decomposition()
        # logger.info("Finished process_decomposition")
        #
        # logger.info("Starting process_step_back_prompting")
        # self.process_step_back_prompting()
        # logger.info("Finished process_step_back_prompting")
        #
        # logger.info("Starting process_hypothetical_document_embeddings")
        # self.process_hypothetical_document_embeddings()
        # logger.info("Finished process_hypothetical_document_embeddings")

        # logger.info("Starting process_hypothetical_document_embeddings")
        # self.process_routing_n_query_construction()
        # logger.info("Finished process_hypothetical_document_embeddings")

        logger.info("Starting process_advanced_indexing_strategies")
        self.process_advanced_indexing_strategies()
        logger.info("Finished process_advanced_indexing_strategies")


if __name__ == "__main__":
    query_ = "What is Task Decomposition?"

    # 3. Create an instance of the RAGChainBuilder
    rag_builder = RAGChainBuilder(query=query_)
    rag_builder.run_pipeline()

    # 4. Use the instance to build and run the chain
    # rag_builder.process_and_create_retriever_v1()
    # rag_builder.process_and_create_retriever_v2_generate_questions()
