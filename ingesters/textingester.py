from langchain_community.document_loaders import TextLoader, PyPDFLoader

from .baseingester import BaseIngester


class TextIngester(BaseIngester):
    def ingest_file(self, file_path: str):
        documents = TextLoader(file_path, autodetect_encoding=True).load()
        self._add_text_to_collection(
            documents[0].page_content,
            None
        )


class PDFIngester(BaseIngester):
    def ingest_file(self, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        for page in pages:
            self._add_text_to_collection(page.page_content, None)
