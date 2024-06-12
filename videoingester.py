from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import OllamaEmbeddings

import ffmpeg

import whisper

from langchain_chroma import Chroma


class VideoIngester:
    def __init__(self, chromadb_client,  collection_name, whisper_model='base', embeddings_model='llama3', chunk_size=1024, chunk_overlap=128):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.whisper_model = whisper.load_model(whisper_model)
        self.vectorstore = Chroma(
            client=chromadb_client,
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(model=embeddings_model),
        )

    @staticmethod
    def _extract_audio_from_video_file(file_path):
        output_filename = f"{file_path.split('.')[-1]}.wav"
        _ = (ffmpeg.input(file_path).
             audio.
             output(output_filename, acodec='pcm_s16le', ac=1, ar='16k').
             overwrite_output().
             run()
             )
        return output_filename

    def _transcribe_audio_file(self, file_path):
        transcript = self.whisper_model.transcribe(file_path)
        return transcript

    def _add_transcript_to_collection(self, transcript_text, transcript_metadata=None):
        splits = self.text_splitter.split_text(transcript_text)
        self.vectorstore.add_texts(
            texts=splits,
            metadatas=[transcript_metadata] * len(splits),
        )

    def ingest_video_file(self, file_path: str):
        audio_filename = self._extract_audio_from_video_file(file_path)
        transcript = self._transcribe_audio_file(audio_filename)
        self._add_transcript_to_collection(
            transcript['text'],
            {"language": transcript['language']}
        )

        return len(transcript['segments'])

    def ingest_text_file(self, file_path: str):
        documents = UnstructuredFileLoader(file_path).load()
        self._add_transcript_to_collection(
            documents[0].page_content,
            None
        )
