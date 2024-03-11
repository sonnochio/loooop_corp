# LOOOOOP Corp.

LOOOOOP Corp. is a sophisticated interactive exhibition framework designed by LOOOOOP to highlight the ethical implications and potential misuse of AI when leveraging non-consensual human data. By simulating an environment where personal information is disguisedly collected and used to generate personalized rumors, the project underscores the significance of privacy, consent, and transparency in AI applications.

The core tech is a RumorGenerator, `main.py`, utilizes advanced AI models to process personal data collected from participants, generating rumors that are both textual and visual. These rumors are then projected in an exhibition setup, engaging visitors through a narrative that seamlessly blends technology with art to provoke thought on digital ethics.

## Features

- **Personalized Rumor Generation**: Dynamically creates rumors based on user input, using AI models.
- **Multimedia Interaction**: Incorporates audio and visual data processing for a rich interactive experience.
- **Privacy and Ethics Exploration**: Simulates the impact of AI misused for personal data, encouraging discussions on digital privacy and ethical AI use.

## Getting Started

### Prerequisites
Ensure you have Python 3.x installed, along with the following packages:
- `pyaudio`
- `opencv-python`
- `Pillow`
- `numpy`
- `moviepy`
- `firebase-admin`
- `python-dotenv`
- `openai`

These can be installed using pip:

```bash
pip install -r requirements
brew install ffmpeg
brew install portaudio
```
### Setup

1. **Download Video**: First, download the video from [Google Drive](https://drive.google.com/file/d/1S3111Ju50ZbsuiWTxmhUVCgJpNT31vai/view?usp=sharing) and save it as `final.mp4` in your project directory.

2. **Resize Video**: To ensure the video fits your screen size for an optimal display during the exhibition, run `resize.py`. This will adjust the video to match your screen's resolution.
    ```bash
    python resize.py
    ```

3. **Environment Configuration**: Set up your environment by creating a `.env` file in the root of your project directory. This file should contain your OpenAI API key:
    ```
    OPENAI_API=your_api_key_here
    ```

4. **Firebase Configuration**: Ensure you have downloaded your Firebase Admin SDK configuration file. Update the path to this file in both `main.py` and `project.py` to match where you've saved it in your project directory.

### Installation Reminder

Before proceeding, make sure you have installed the necessary system dependencies:

- **FFmpeg** for video and audio processing:
    ```bash
    brew install ffmpeg
    ```
    
- **PortAudio** for audio capture and playback functionalities:
    ```bash
    brew install portaudio
    ```
    
- **Matplotlib** for generating visuals:
    ```bash
    pip3 install matplotlib
    ```
Ensure these commands are executed in your terminal to set up the project environment correctly.
