# Speech AI Tutorial

This tutorial walks you through building speech-enabled applications with Thunders AI. You'll learn how to convert text to natural-sounding speech, transcribe audio recordings, build a voice assistant, and handle real-time audio streams. The Speech AI module supports 50+ languages, custom voice selection, and both file-based and streaming audio processing.

## What You'll Build

A voice assistant application that:
- Converts text to speech with customizable voices and speaking rates
- Transcribes audio files with word-level timestamps
- Supports multiple languages for both TTS and STT
- Processes real-time audio streams for live transcription

## Prerequisites

- Python 3.9+
- Thunders AI with speech extras: `pip install thunders-ai[speech]`
- A microphone (for real-time features)
- Audio sample files for testing

## Step 1: Text-to-Speech Basics

Convert written text into natural-sounding audio:

```python
from thunders_ai import ThundersAI

ai = ThundersAI()

# Basic TTS
result = ai.tts("Welcome to Thunders AI, the future of intelligent systems.")
print(f"Generated {result.duration:.1f}s of audio at {result.sample_rate}Hz")

# Save the audio to a file
with open("welcome.wav", "wb") as f:
    f.write(result.audio_bytes)

# Or use the pre-saved path
print(f"Audio saved to: {result.audio_path}")
```

## Step 2: Voice Customization

Choose from multiple voice options and adjust speaking parameters:

```python
# List available voices
voices = ai.get_available_voices()
for voice in voices:
    print(f"  {voice['id']}: {voice['name']} ({voice['language']}, {voice['gender']})")

# Use a specific voice
result = ai.tts(
    "Hello! I'm Aria, your AI assistant.",
    voice="aria",
    rate=1.0,  # Normal speed
)

# Speed up for skimmable content
result = ai.tts(
    "Here's a quick summary of the latest research papers.",
    voice="aria",
    rate=1.5,  # 50% faster
)

# Slow down for emphasis or accessibility
result = ai.tts(
    "Please listen carefully to these important instructions.",
    voice="aria",
    rate=0.8,  # 20% slower
)
```

## Step 3: Speech-to-Text Transcription

Transcribe audio recordings into text with confidence scores and timestamps:

```python
# Transcribe an audio file
result = ai.stt("meeting_recording.wav")

print(f"Transcription: {result.text}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Detected language: {result.language}")

# Access word-level segments with timestamps
for segment in result.segments:
    print(f"  [{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
```

## Step 4: Multi-Language Support

Thunders AI supports transcription and synthesis in 50+ languages:

```python
# Transcribe Spanish audio
result = ai.stt("entrevista.mp3", language="es")
print(f"Transcripción: {result.text}")

# Generate speech in Japanese
result = ai.tts("AIの未来へようこそ", language="ja", voice="japanese-female-1")

# List all supported languages
languages = ai.get_supported_languages()
print(f"Supported languages: {', '.join(sorted(languages))}")

# Common languages include: en, es, fr, de, zh, ja, ko, pt, ru, ar, hi, it, nl, pl, tr
```

## Step 5: Building a Voice Assistant

Combine TTS and STT to create an interactive voice assistant:

```python
"""Voice assistant built with Thunders AI Speech and Chat modules."""
from thunders_ai import ThundersAI, ThundersConfig

def voice_assistant():
    config = ThundersConfig(
        model="thunders-7b",
        system_prompt="You are a helpful voice assistant. Keep responses brief and conversational."
    )
    ai = ThundersAI(config=config)

    print("🎙️ Voice Assistant (say 'quit' to exit)")
    print("Listening...\n")

    while True:
        # Record and transcribe user speech
        # (In production, use a real audio capture library like PyAudio)
        result = ai.stt("user_input.wav")
        user_text = result.text

        if not user_text or user_text.lower() in ("quit", "exit"):
            break

        print(f"You: {user_text}")

        # Generate AI response
        response = ai.chat(user_text)
        print(f"Assistant: {response}")

        # Convert response to speech
        audio = ai.tts(response, voice="aria")
        with open("assistant_response.wav", "wb") as f:
            f.write(audio.audio_bytes)

        # In production, play the audio file here
        print("(Audio response saved to assistant_response.wav)\n")

    print("Goodbye!")

if __name__ == "__main__":
    voice_assistant()
```

## Step 6: Real-Time Streaming Transcription

For live transcription of continuous audio, use streaming STT:

```python
import pyaudio

ai = ThundersAI()
chunk_duration = 2  # Process in 2-second chunks

# Set up audio capture
pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=16000 * chunk_duration,
)

print("Live transcription started. Press Ctrl+C to stop.")
try:
    while True:
        audio_data = stream.read(16000 * chunk_duration)
        result = ai.stt(audio_data, language="en")
        if result.text.strip():
            print(f"[{result.confidence:.0%}] {result.text}")
except KeyboardInterrupt:
    print("\nStopped.")

stream.stop_stream()
stream.close()
pa.terminate()
```

## Audio Format Support

| Format | TTS Output | STT Input | Notes |
|--------|-----------|-----------|-------|
| WAV | ✅ | ✅ | Default format, lossless |
| MP3 | ✅ | ✅ | Compressed, smaller files |
| OGG | ✅ | ✅ | Open format, good compression |
| FLAC | ❌ | ✅ | Lossless compression |
| WebM | ❌ | ✅ | Web-friendly format |

## Best Practices

- **Use language hints** when you know the audio language — it improves STT accuracy by 10-20%
- **Prefer WAV for STT input** — uncompressed audio gives the best transcription accuracy
- **Adjust speaking rate** for different audiences (slower for accessibility, faster for power users)
- **Cache TTS results** for frequently spoken phrases to reduce latency and cost
- **Use streaming** for real-time applications to avoid waiting for full audio generation

## Next Steps

- Combine speech with **vision AI** for multimodal interactions
- Build a **podcast transcription tool** with speaker diarization
- Create **accessibility features** like screen readers using TTS
- Deploy as an **API service** for integration with telephony systems
