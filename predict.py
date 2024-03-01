# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from whisperspeech.pipeline import Pipeline

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

    def predict(
        self,
        prompt: str = Input(description="Text to synthesize", default="This is the first demo of Whisper Speech, a fully open source text-to-speech model trained by Collabora and Lion on the Juwels supercomputer."),
        language: str = Input(
            description="Language to synthesize", default="en",
            choices=["en", "pl"]
        ),
        speaker: str = Input(
            description="URL of an OGG audio file for zero-shot voice cloning. (ex: https://upload.wikimedia.org/wikipedia/commons/7/75/Winston_Churchill_-_Be_Ye_Men_of_Valour.ogg)",
            default="",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        output_path = "/tmp/output.wav"
        # Check if voice cloning is used
        if(speaker == ""):
            self.pipe.generate_to_file(output_path, prompt, lang=language)
        else:
            self.pipe.generate_to_file(output_path, prompt, lang=language, speaker=speaker)
        return Path(output_path)
