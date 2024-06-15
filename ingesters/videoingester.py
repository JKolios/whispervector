import ffmpeg

import whisper


from .baseingester import BaseIngester

class VideoIngester(BaseIngester):
    def __init__(self, vectorstore, whisper_model='base'):
        super(VideoIngester, self).__init__(vectorstore)
        self.whisper_model = whisper.load_model(whisper_model)

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

    def ingest_file(self, file_path: str):
        audio_filename = self._extract_audio_from_video_file(file_path)
        transcript = self._transcribe_audio_file(audio_filename)
        self._add_text_to_collection(
            transcript['text'],
            {"language": transcript['language']}
        )
