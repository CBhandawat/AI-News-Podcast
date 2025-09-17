# AI-News-Podcast
Your AI Gossiper

## Features

📥 Scrape AI News — fetch the latest AI updates.

📰 Generate Podcast Summary — turns news into podcast-style narration.

🔊 Convert to Audio — choose between:

--Sarvam AI

--ElevenLabs

--VibeVoice

🎥 Animate Talking Head — generate a video host reading the news using MuseTalk.

## Installation

### Clone this repository
```
git clone https://github.com/CBhandawat/AI-New-Podcast.git
cd AI-News-Podcast
```


### Install dependencies
```
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Install MuseTalk
```
git clone https://github.com/TMElyralab/MuseTalk.git
```
Make sure MuseTalk/ sits in the same directory as app.py.

After cloning MuseTalk, make sure you follow all the steps mentioned in MuseTalk repo README (https://github.com/TMElyralab/MuseTalk)

### Install VibeVoice
```
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

## MuseTalk Files I changed:

1. MuseTalk/musetalk/utils/preprocessing.py
2. MuseTalk/musetalk/utils/face_parsing/resnet.py
3. MuseTalk/scripts/inference.py

## Running the App

### Start the Gradio app:
```
python app.py
```

### Open in browser:
```
http://localhost:7861/
```


