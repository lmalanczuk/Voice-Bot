# VoiceAgent - Local Voice Assistant based on Whisper + LLaMA + TTS

VoiceAgent is a fully local and offline assistant that:
- listens for commands
- transcribes speech using whisper
- generates responses via local LLM (GGUF, llama.cpp)
- speaks responses using system tts (pyttsx3)
- automatically saves notes

## Features
- PL speech recognition (Whisper)
- Local LLM support (Bielik, Mistral, LLaMA - GGUF)
- TTS via Windows system voices (SAPI5)
- Automatic notes creation (WiP)
- Fully offline operation

## Required dependencies
- speechrecognition
- pyaudio
- pyttsx3
- faster-whisper
- llama-cpp-python
