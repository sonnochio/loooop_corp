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
import numpy as np
import queue
from moviepy.editor import VideoFileClip
import threading
import time
from functools import partial
import io
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime

cred = credentials.Certificate("./fb_secret.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'loooop-corp.appspot.com'  # Update this to your Cloud Storage bucket name
})


db = firestore.client()
bucket = storage.bucket()

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
video_path="final_resized.mp4"
api_key=os.environ["OPENAI_API"]
print(api_key)

client = OpenAI(api_key=api_key)

#====Define video player====
# Placeholder for a condition check, replace with your actual condition
def condition_to_play_video():
    # This could be any logic: a response from a background thread, user input, etc.
    time.sleep(5)  # Simulate waiting for a condition
    return True

def play_video_segment(video_path, start_time, end_time):
    """Plays a specific segment of a video."""
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    clip.preview(fps=24, audio=True)  # Adjust FPS to match your video

def play_video_segment_with_sound(video_path, start_time, end_time):
    video = VideoFileClip(video_path).subclip(start_time, end_time)
    video.preview()

##==========Find the right camera to use==========
def find_usb_camera(max_index=10):
    # Try to open a camera at each index from 0 to max_index and see the view
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue  # If the camera at the current index did not open, move to the next index

        # If the camera opens, attempt to read a frame to verify it works
        ret, frame = cap.read()
        if ret:
            print(f"Camera found at index {index}")
            # Display the frame to verify it's the correct camera
            cv2.imshow(f"Camera Index {index}", frame)
            cv2.waitKey(5000)  # Wait for 1 second so you can see the frame
            cv2.destroyAllWindows()
        cap.release()

# find_usb_camera()
# print(api_key)
# breakpoint()

##==========Starting Trigger==========

#This section activates the program when a face is detected
#activation of program using openCV

def face_present():
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start the webcam
    cap = cv2.VideoCapture(0)
    
    yes_counter = 0
    attempt_counter = 0
    start_time = time.time()
    
    while True:
        # Read frames from the webcam
        ret, frame = cap.read()
    
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
        # Increase attempt counter
        attempt_counter += 1
    
        # Check if faces were detected
        if len(faces) > 0:
            yes_counter += 1  # Increase "Yes" counter if faces are detected
    
        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        # # Display the frame
        # cv2.imshow('Face Detection', frame)
    
        # Break the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        # Control the detection frequency to 10 times per second
        time.sleep(max(0, 0.1 - (time.time() - start_time)))
        start_time = time.time()
    
        # Check if 10 attempts have been made
        if attempt_counter == 20:
            # Print "There is a face" if 8 or more detections were "Yes"
            if yes_counter >= 16:
                # Release the VideoCapture object and close the windows
                # cap.release(0)
                # cv2.destroyAllWindows()
                return True 
            # Reset counters
            else:
                yes_counter = 0
                attempt_counter = 0
    
    
##==========Recording & Transcribing Voice==========

def start_recording(num, recording_seconds=5.0):  # Ensure recording_seconds is a float
    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = float(recording_seconds)  # Convert to float to ensure correct operation

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Recording...")
    
    frames = []
    total_frames = int(RATE / CHUNK * RECORD_SECONDS)  # Calculate total frames to capture
    for _ in range(total_frames):
        data = stream.read(CHUNK)
        frames.append(data)
    
    # Capture any remaining part of the recording period not covered by full chunks
    remaining_frames = int((RATE * RECORD_SECONDS) % CHUNK)
    if remaining_frames > 0:
        data = stream.read(remaining_frames)
        frames.append(data)
    
    print("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    WAVE_OUTPUT_FILENAME = f"./audio/test_{num}.wav"
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Recording saved as {WAVE_OUTPUT_FILENAME}")
    
    return WAVE_OUTPUT_FILENAME


def stt (num, recording_seconds=5):
    start_recording(num,recording_seconds)
    file_name=f"./audio/test_{num}.wav"
    audio_file= open(file_name, "rb")
    transcript = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file)

    return (file_name,transcript)


##==========Text Rumor Creation==========
#system question prompt to users 

prompts="""What’s your name?|What do you do for a living?|Can you describe a moment in your life where you felt completely at peace or truly happy? (5 - second)|Can you describe a challenge you've recently faced and how you approached it?|What's one piece of advice you've received that has profoundly impacted your life? How have you applied it?|How much did you spend on the last piece of clothing?|( Checkout section ) what’s your address?"""
prompt_ls=prompts.split('|')

def create_rumors(db):
    rumour_system_promt=f"""I am working on an art project. Based on the questions and answers from users below , can you generate some shocking news that are untrue about this person. or alter the answers to generate rumors that are opposite to what they say. You should generate 5 facts or news or rumours that demonstrate using this person's data against them and generate untrue information. The 5 facts should be 5 sentences that are 10 - 20 words each. 

You should output the 5 sentences in the following format :

"art project starts"|sentence 1| sentence 2|sentence 3|sentence 4|sentence 5

Below are the list of questions and answers:  

"""

    
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": f"{rumour_system_promt}"},
        {"role": "user", "content": f"{db}"},
      ]
    )
    rumors=response.choices[0].message.content.split('|')[1:]
    print(rumors)
    return rumors
# create_rumors(db)

##==========Image Rumor Creation ==========


def take_picture_with_webcam(id):
    # Open a connection to the webcam (0 is usually the default webcam)
    
    cap = cv2.VideoCapture(0)

 
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return None

    cv2.waitKey(500)

    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Release the webcam connection
    cap.release()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Unable to capture frame.")
        return None

    # Define the directory to save pictures
    pictures_dir = "pictures"
    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir)

    # Generate a unique identifier for the picture filename
    while os.path.exists(os.path.join(pictures_dir, f"{id}.png")):
        id += 1

    # Save the captured frame as an image
    picture_path = os.path.join(pictures_dir, f"{id}.png")
    

    
    cv2.imwrite(picture_path, frame)

    #crop to be square
    with Image.open(picture_path) as img:
        width, height = img.size

        # Determine the size of the square (use the smaller dimension)
        new_size = min(width, height)

        # Calculate the left, upper, right, and lower pixels to crop
        left = (width - new_size)/2
        top = (height - new_size)/2
        right = (width + new_size)/2
        bottom = (height + new_size)/2

        # Perform the crop
        img_cropped = img.crop((left, top, right, bottom))
        crop_path = f"./crop/{id}.png"
        img_cropped.save(crop_path)
        img_cropped.show()

    print(f"Cropped picture saved successfully: {crop_path}")

    return img_cropped, crop_path

def create_face_mask(image_path, id):
    # Load the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Create an all-black mask
    mask = np.zeros((h, w), dtype="uint8")

    # Loop over the faces and draw white rectangles on the mask
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 255

    # Create a 4-channel image (BGRA) by adding the mask as the alpha channel
    image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_with_alpha[:, :, 3] = mask

    # Save the result
    output_path = f'./masks/{id}.png'
    cv2.imwrite(output_path, image_with_alpha)

    crop_mask_path=f'./crop_mask/{id}.png'
    
    with Image.open(output_path) as img:
        width, height = img.size

        # Determine the size of the square (use the smaller dimension)
        new_size = min(width, height)

        # Calculate the left, upper, right, and lower pixels to crop
        left = (width - new_size)/2
        top = (height - new_size)/2
        right = (width + new_size)/2
        bottom = (height + new_size)/2

        # Perform the crop
        img_cropped = img.crop((left, top, right, bottom))
        crop_path = f"./crop_mask/{id}.png"
        img_cropped.save(crop_mask_path)
        img_cropped.show()
        
    return crop_mask_path
    
    #rumorimage prompt creation 
def create_rumor_image_prompt(rumors):
    rumor_image_prompt_response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are an AI assistant. Your job is to create a prompt for the image generating AI model Dalle-2 based on the user content. Your output should be useful to generate a picture. You can pick one setence or the all the sentences to focus on from the user content, in order to create a compelling image story. You should specify a place, a time, a color scheme, the emotions, the texture. the style of the image should be realistic "},
        {"role": "user", "content": f"{rumors}"},
  ]
    )
    rumor_image_prompt=rumor_image_prompt_response.choices[0].message.content

    return rumor_image_prompt

def summarize_image_prompt(rumor_image_prompt):
    summary_prompt_response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "you are an AI assist, you job is to summarize the user prompt into 1 sentence that describes the story within. It should always contain the person's name, the action and the context. This should be between 10-20 words."},
        {"role": "user", "content": f"{rumor_image_prompt}"},
  ]
    )
    print(summary_prompt_response)
    summary_prompt_response=summary_prompt_response.choices[0].message.content
    print(summary_prompt_response)
    return summary_prompt_response


#===========Threading STT and video playing==========

def stt_background_task(num, recording_seconds, callback=None):
    """Simulates an STT processing task."""
    # Here, integrate your STT processing logic

    start_recording(num,recording_seconds)
    file_name=f"./audio/test_{num}.wav"
    audio_file= open(file_name, "rb")
    transcript = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file)
    audio_file.close()  # Make sure to close the file after reading

    transcription = transcript.text

    
    # Once processing is complete, use a callback to handle the result
    if callback:
        callback(transcription)


def handle_stt_result(db, prompt, num,transcription):
    """Handles the STT result by appending it to the db list."""
    db.append(f"Q{num}:{prompt},A:{transcription}")
    print("\n", "🫵", transcription, "\n", "--------------------------------------------------------", "\n")


#=====Upload Data Point=====
def upload_image_to_storage(image, image_name):
    """Uploads a PIL.Image to Cloud Storage."""
    # Convert PIL.Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')  # You can change 'JPEG' to another format if needed
    img_byte_arr = img_byte_arr.getvalue()

    # Create a new blob in Cloud Storage
    blob = bucket.blob(f'images/{image_name}')  # You can customize the path as needed
    blob.upload_from_string(img_byte_arr, content_type='image/jpeg')

    # Make the blob publicly viewable
    blob.make_public()

    # Return the public URL
    return blob.public_url

def upload_data_point(rumors, summarized_image_prompt, image_url, blob_url):
    doc_ref = db.collection('data_points').document()  # Creates a new document
    doc_ref.set({
        'rumors': rumors,
        'summarized_image_prompt': summarized_image_prompt,
        'blob_url':blob_url,
        'image_url': image_url,
        'timestamp':firestore.SERVER_TIMESTAMP
    })
    print("Data point uploaded successfully.")


#=====Backgroun Video Loop=====
def play_background_loop():
    """Function to loop a background video segment."""
    while not face_present():
        play_video_segment(video_path, 121.9, 123.0)
        time.sleep(0.1)  # Short sleep to prevent hogging CPU resources


##=====Main Program Logic=====
def main_program_logic():

    timeline={    0: ['15.9', '16.67', '16.7', '20.73', 4.02],
    1: ['20.77', '21.67', '21.7', '28.00', 6.65],
    2: ['28.03', '32.57', '32.6', '47.23', 14.82],
    3: ['47.27', '51.00', '51.03', '67.60', 16.28],
    4: ['67.63', '73.67', '73.7', '89.23', 15.77],
    5: ['89.27', '91.50', '91.53', '97.53', 6.00],
    6: ['97.57', '98.30', '98.33', '104.73', 6.2]}

    db=[]
    id=1
    #play intro
    #'01:03:05.35'
    play_video_segment(video_path, 0.00, 15.8)
    for num,prompt in enumerate(prompt_ls):
        
        callback_with_context = partial(handle_stt_result, db, prompt, num)

        # Play the first specified segment
        play_video_segment(video_path, timeline[num][0], timeline[num][1])
        
        # Start STT processing in the background for this segment
        stt_thread = threading.Thread(target=stt_background_task, args=(num, timeline[num][4], callback_with_context))
        stt_thread.start()
        
        # After starting STT processing, immediately play the next segment
        play_video_segment(video_path, timeline[num][2], timeline[num][3])
        
        # Assuming you wait for the STT thread if necessary
        stt_thread.join()

    play_video_segment(video_path, 104.74, 123.40)    
        
    #creates rumors
    rumors=create_rumors(db)
    
    #takes picture
    captured_image, image_path = take_picture_with_webcam(id=1)
    print("Image path:", image_path)
    
    #Import image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def create_image(rumor_image_prompt):
    
        response = client.images.edit(
        model="dall-e-2",
        image=open(image_path, "rb"),
        mask=open(masked_image_path, "rb"),
        prompt=rumor_image_prompt,
        n=1,
        size="512x512"
        )
        image_url = response.data[0].url
        return image_url


    #generate mask
    masked_image_path = create_face_mask(image_path,1)
    print(f"Masked image saved to: {masked_image_path}")

    #create rumor image prompts 
    rumor_image_prompt=create_rumor_image_prompt(rumors)
    summarized_image_prompt=summarize_image_prompt(rumor_image_prompt)
    print(summarized_image_prompt)

    #create rumor image
    image_url=create_image(rumor_image_prompt)

        #open rumor image
    response_rumor = requests.get(image_url)
    img_rumor = Image.open(BytesIO(response_rumor.content))
    

    blob_url=upload_image_to_storage(img_rumor, image_name=f'{datetime.now()}.jpg')

    upload_data_point(rumors, summarized_image_prompt, image_url, blob_url)





##==========TOTAL TEST==========
#different recording secs for diff questions
#prompt with some example answers
def main():
    try:
        while True:
            print("Checking for face presence...")
            while not face_present():
                # Optional: Sleep to reduce CPU usage, adjust the time as needed
                time.sleep(1)
            
            print("Face detected. Executing program logic.")
            # Execute the main program logic after a face is detected
            main_program_logic()
            
            # After completing main_program_logic, the program will loop back
            # and start checking for a face again
            
    except KeyboardInterrupt:
        print("Program terminated by user.")

    

if __name__ == '__main__':
    main()
   
    
    