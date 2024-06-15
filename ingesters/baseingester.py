from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 512
CHUNK_OVERLAP = 128


class BaseIngester:
    def __init__(self, vectorstore):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.vectorstore = vectorstore

    def _add_text_to_collection(self, text, text_metadata=None):
        splits = self.text_splitter.split_text(text)
        self.vectorstore.add_texts(
            texts=splits,
            metadatas=[text_metadata] * len(splits),
        )

    def ingest_file(self, file_path):
        pass
