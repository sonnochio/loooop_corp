import pyaudio
import wave
import cv2
import time
from openai import OpenAI
from matplotlib import pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path
import os


dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

api_key=os.environ["OPENAI_API"]
print(api_key)

client = OpenAI(api_key=api_key)