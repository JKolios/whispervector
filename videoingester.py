from langchain.text_splitter import RecursiveCharacterTextSplitter

import ffmpeg

import whisper

import chromadb


class VideoIngester:
    def __init__(self, collection_name):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.whisper_model = whisper.load_model("base")
        self.chromadb_client = chromadb.PersistentClient(path="./chromadb")
        self.collection_name = collection_name
        self.document_collection = self.chromadb_client.get_or_create_collection(collection_name)

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

    def _add_transcript_to_collection(self, transcript_id, transcript_text, transcript_metadata=None):
        self.document_collection = self.chromadb_client.get_or_create_collection(self.collection_name)
        self.document_collection.add(
            documents=[transcript_text],
            metadatas=[transcript_metadata],  # filter on these!
            ids=[transcript_id],
        )

    def ingest(self, file_path: str, file_name: str):
        audio_filename = self._extract_audio_from_video_file(file_path)
        transcript = self._transcribe_audio_file(audio_filename)
        self._add_transcript_to_collection(
            file_name,
            transcript['text'],
            {"language": transcript['language']}
        )

        return len(transcript['segments'])



