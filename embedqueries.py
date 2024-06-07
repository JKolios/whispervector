import chromadb


class EmbedQueries:
    def __init__(self):
        self.chromadb_client = chromadb.PersistentClient(path="./chromadb")
        self.document_collection = self.chromadb_client.get_or_create_collection("transcripts")

    def query(self, query_text, n_results=5):
        results = self.document_collection.query(query_texts=[query_text], n_results=n_results)
        return results['documents']