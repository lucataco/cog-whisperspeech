# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

WEIGHTS_FOLDER = "/src/models/"
os.environ['HF_HOME'] = WEIGHTS_FOLDER 
os.environ['HF_HUB_CACHE'] = WEIGHTS_FOLDER
os.environ['TORCH_HOME'] = WEIGHTS_FOLDER
os.environ['PYANNOTE_CACHE'] = WEIGHTS_FOLDER

from cog import BasePredictor, Input, Path
import torch
from whisperspeech.pipeline import Pipeline
from speechbrain.pretrained import EncoderClassifier


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

        # source: https://github.com/collabora/WhisperSpeech/blob/a4f9c2de1a7e2e0b77f2acb08374de347414e2fa/whisperspeech/pipeline.py#L68-L72
        self.pipe.encoder = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                                          savedir=Path(WEIGHTS_FOLDER) / "speechbrain",
                                                          run_opts={"device": "cuda"})
 
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
