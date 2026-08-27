# Dialogue Frame Finder — Approach

## 1. Problem Understanding

The objective of this project is to locate the exact point in a video where a given dialogue first occurs and return the corresponding video frame, timestamp, detected text, confidence, and saved frame image.

The system accepts:

1. A YouTube video URL
2. A target dialogue/text

The system then searches the video using multiple sources of evidence:

- Subtitle tracks
- Spoken audio
- On-screen text
- Video frame timestamps

The goal is not simply to find the dialogue somewhere in the video. The goal is to locate the most accurate frame and timestamp associated with the dialogue.

---

# 2. Core Idea

The main idea is to use a **multi-modal search pipeline** instead of relying on a single technique.

A video can contain dialogue in different forms:

- The dialogue may exist as a subtitle track.
- The dialogue may only be spoken in the audio.
- The dialogue may be visibly displayed on the screen.
- The dialogue may exist in both speech and on-screen subtitles.

Therefore, the system combines:

```text
Subtitle Search
       +
Speech Recognition
       +
Visual OCR
       ↓
Candidate Location
       ↓
Fine Frame Search
       ↓
Final Frame Resolution