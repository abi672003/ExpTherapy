#!/usr/bin/env python3
"""
🎤 Speech Fluency & Disfluency Analysis (Streamlit App)

Upload an audio file and this app will:
- Detect repeated words (stuttering/nervousness)
- Detect filler words ("um", "uh", etc.)
- Detect long pauses (hesitation)
- Compute average pause duration and speech rate

Requirements:
-------------
- streamlit
- openai-whisper
- torch
- librosa
"""

import streamlit as st
import whisper
import numpy as np
from pathlib import Path
import tempfile

# Filler words
FILLERS = {"um", "uh", "erm", "hmm", "you know", "like"}

# Fluency analysis logic
def analyze_fluency(audio_path: str):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for seg in result["segments"]:
        if "words" in seg:
            words.extend(seg["words"])

    repetitions = 0
    filler_count = 0
    pauses = []
    prev_word, prev_end = None, None

    transcript_log = []

    for w in words:
        word = w["word"].lower().strip()
        start, end = w["start"], w["end"]

        # Filler words
        if word in FILLERS:
            filler_count += 1
            transcript_log.append(f"🟡 Filler detected: **{word}**")

        # Repetitions
        if prev_word and word == prev_word:
            repetitions += 1
            transcript_log.append(f"🔴 Repetition detected: **{word}**")

        # Pauses
        if prev_end is not None:
            pause = start - prev_end
            if pause > 0.7:  # more than 0.7s silence
                pauses.append(pause)
                transcript_log.append(f"⏸️ Pause of {pause:.2f}s detected")

        transcript_log.append(word)
        prev_word, prev_end = word, end

    # Summary metrics
    total_words = len(words)
    duration = result["segments"][-1]["end"] if result["segments"] else 0
    speech_rate = total_words / duration if duration > 0 else 0

    metrics = {
        "repetitions": repetitions,
        "fillers": filler_count,
        "avg_pause": (sum(pauses) / len(pauses)) if pauses else 0,
        "speech_rate": speech_rate,
        "total_words": total_words,
    }

    return transcript_log, metrics


# ================== Streamlit App ==================
st.set_page_config(page_title="Speech Fluency Analyzer", page_icon="🎤", layout="centered")
st.title("🎤 Speech Fluency & Disfluency Analysis")

st.write("Upload an audio file and get insights on fillers, pauses, repetitions, and speech rate.")

uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/mp3")

    with st.spinner("Analyzing speech... ⏳"):
        transcript_log, metrics = analyze_fluency(tmp_path)

    st.subheader("📝 Transcript with Annotations")
    st.write(" ".join(transcript_log))

    st.subheader("📊 Speaking Style Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Repetitions", metrics["repetitions"])
    col2.metric("Filler Words", metrics["fillers"])
    col3.metric("Total Words", metrics["total_words"])

    col4, col5 = st.columns(2)
    col4.metric("Avg Pause (s)", f"{metrics['avg_pause']:.2f}" if metrics["avg_pause"] > 0 else "No pauses")
    col5.metric("Speech Rate (w/sec)", f"{metrics['speech_rate']:.2f}")

    st.success("✅ Analysis Complete")
else:
    st.info("Please upload an audio file to start analysis.")
