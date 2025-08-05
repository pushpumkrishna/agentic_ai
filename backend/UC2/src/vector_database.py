# Our custom vector database
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.utils.constants import COMPANY_INFORMATION


class VectorDatabase:
    def __init__(self):
        # Store all vectors in an array
        self.vectors = []

    # Add vector to database
    def add_vector(self, vec_id, vector, metadata=None):
        record = {
            "id": vec_id,
            "vector": np.array(vector, dtype=np.float32),
            "metadata": metadata,
        }

        self.vectors.append(record)

    # Retrieve all vectors from database
    def get_all_vectors(self):
        return self.vectors

    # Calculate cosine similarity between vectors
    @staticmethod
    def _cosine_similarity(vector_a, vector_b):
        # Calculate dot product
        dot_product = np.dot(vector_a, vector_b)

        # Calculate the magnitude of vector A
        magnitude_a = np.linalg.norm(vector_a)

        # Calculate the magnitude of vector B
        magnitude_b = np.linalg.norm(vector_b)

        cosine_similarity = dot_product / (
            magnitude_a * magnitude_b + 1e-8
        )  # small epsilon to avoid division by zero

        return cosine_similarity

    # Search for similar vectors and return the top_k results
    def search(self, query_vector, top_k=3):
        query_vector = np.array(query_vector, dtype=np.float32)

        # Stores the top_k results
        results = []

        for record in self.vectors:
            sim = self._cosine_similarity(query_vector, record["vector"])

            results.append(
                {"id": record["id"], "similarity": sim, "metadata": record["metadata"]}
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    @staticmethod
    def create_db():
        # Set up our custom vector database
        company_db = VectorDatabase()

        return company_db

    def populate(self):
        # Populate the custom vector database
        # Embedding model
        model = SentenceTransformer(
            "C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/sentence_transformer/"
        )
        company_db = self.create_db()
        for idx, sentence in enumerate(COMPANY_INFORMATION):
            # Create sentence embedding
            embedding = model.encode(sentence)

            # Add sentence embedding to the database

            company_db.add_vector(
                vec_id=f"sentence_{idx}",
                vector=embedding,
                metadata={"sentence": sentence},
            )

        # for record in company_db.get_all_vectors():
        #     print(record)
        #     break
        #
        # # Check embedding size of a record
        # for record in company_db.get_all_vectors():
        #     print(record["vector"].shape)
        #     break

        print("Number of records in the database: ", len(company_db.get_all_vectors()))

        return model, company_db
