# WhisperSpeech Cog Model

This is an implementation of [WhisperSpeech](https://github.com/collabora/WhisperSpeech) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Basic Usage

for prediction,

    cog predict -i prompt="This is the first demo of Whisper Speech, a fully open source text-to-speech model trained by Collabora and Lion on the Juwels supercomputer"

[output](output.wav)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. [WhisperSpeech/WhisperSpeech](https://huggingface.co/WhisperSpeech/WhisperSpeech) is released under the [MIT](https://opensource.org/licenses/MIT) license.
