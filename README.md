# MoodLens — Real-time Emotion Detector + Spotify Suggester

Detects facial emotions in real time via webcam or photo upload, then recommends Spotify playlists matched to your mood.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)

---

## Features

| Feature | Detail |
|---|---|
| Real-time webcam inference | ~30 fps display, detection every 1.5 s |
| 7-class emotion recognition | angry · disgust · fear · happy · sad · surprise · neutral |
| Bounding box overlay | Live annotated video with emotion label + confidence |
| Spotify integration | Fetches 4 mood-matched playlists via public API |
| Photo upload fallback | Works without a webcam |
| Adjustable confidence | Sidebar slider to tune sensitivity |

---

## Quick start

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/moodlens.git
cd moodlens

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

> **Note (slower machines):** in `src/emotion_detector.py` the `mtcnn=False` flag is already set. For even faster inference, swap FER for MediaPipe Face Mesh (see below).

---

## Spotify setup (optional)

1. Go to [Spotify for Developers](https://developer.spotify.com/dashboard) and create a free app.
2. Copy your **Client ID** and **Client Secret**.
3. Paste them into the sidebar when the app is running.

No user login required — the app uses the Client Credentials flow (read-only, public playlists only).

---

## Project structure

```
moodlens/
├── app.py                   # Streamlit UI + webcam loop
├── requirements.txt
├── src/
│   ├── emotion_detector.py  # FER wrapper → returns emotion + bounding box
│   └── spotify_suggester.py # Spotify API client (token cache + search)
└── README.md
```

---

## How it works

```
Webcam frame (BGR)
       │
       ▼
  cv2.cvtColor → RGB
       │
       ▼
  FER.detect_emotions()
  ┌────────────────────────────┐
  │  Haar cascade face detect  │
  │  → CNN (FER-2013 trained)  │
  │  → softmax over 7 classes  │
  └────────────────────────────┘
       │
       ▼
  dominant emotion + confidence
       │
       ├──► annotate frame (cv2.rectangle + putText)
       │
       └──► Spotify search query
                  │
                  ▼
            /v1/search?type=playlist
                  │
                  ▼
            4 playlist links displayed
```

---

## Extending the project

| Idea | Complexity | Impact |
|---|---|---|
| Swap FER for **MediaPipe** + custom ONNX model | Medium | Faster, no TF dependency |
| Add **Grad-CAM** heatmap on face crop | Medium | Shows model explainability |
| Log emotion timeline → **matplotlib chart** | Easy | Visual analytics |
| Deploy to **Hugging Face Spaces** | Easy | Public demo link for CV |
| Add **multi-face** tracking with IDs | Hard | Crowd/meeting analytics angle |

---

## Key dependencies

- [FER](https://github.com/justinshenk/fer) — Facial Emotion Recognition library
- [OpenCV](https://opencv.org/) — video capture & image annotation
- [Streamlit](https://streamlit.io/) — web UI
- [Spotify Web API](https://developer.spotify.com/documentation/web-api) — playlist search

---

## License

MIT — free to use, modify, and include in your portfolio.
