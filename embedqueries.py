
import json


class EmbedQueries:
    def __init__(self, collection_name, chromadb_client):
        self.collection = collection_name
        self.chromadb_client = chromadb_client
        self.document_collection = self.chromadb_client.get_or_create_collection(self.collection)

    def chroma_query(self, query_text, n_results=5):
        results = self.document_collection.query(query_texts=[query_text], n_results=n_results)
        return json.dumps(results)

    def clear_collection(self):
        self.chromadb_client.delete_collection(self.collection)
        self.document_collection = self.chromadb_client.get_or_create_collection(self.collection)

