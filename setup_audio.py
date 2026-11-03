"""Download and check the configured local speech model before serving requests."""
from audiovisual import speech_model

if __name__ == "__main__":
    speech_model()
    print("Audio transcription model is ready.")
