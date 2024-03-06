import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
from firebase_admin import firestore,credentials
import firebase_admin
import io
import queue
import requests
import time
from PIL import Image
import matplotlib.pyplot as plt
from firebase_admin import firestore

# Initialize Cloud Firestore (Assuming Firebase Admin SDK has been initialized)


cred = credentials.Certificate('fb_secret.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

def fetch_latest_data_point():
    # Fetch the most recent document from 'data_points' collection
    docs = db.collection('data_points').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()

    latest_data = None
    for doc in docs:
        latest_data = doc.to_dict()
    return latest_data

def display_data_graphics(data):
    # Fetch image from URL
    response = requests.get(data['image_url'])
    image = Image.open(io.BytesIO(response.content))

    # Set up the plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left column: Display all text in rumors
    rumors_text = "\n".join(data['rumors'])
    axs[0].text(0.5, 0.5, rumors_text, verticalalignment='center', horizontalalignment='center', fontsize=12, wrap=True)
    axs[0].axis('off')

    # Right column: Display the image at the top and the summarized_image_prompt at the bottom
    axs[1].imshow(image)
    axs[1].text(0.5, -0.1, data['summarized_image_prompt'], verticalalignment='top', horizontalalignment='center', transform=axs[1].transAxes, fontsize=12, wrap=True)
    axs[1].axis('off')

    plt.show()

def main():
    data_queue = queue.Queue(maxsize=10)  # Initialize a queue to store the data points

    while True:
        latest_data = fetch_latest_data_point()
        if latest_data is not None:
            # Add the latest data point to the queue
            if not data_queue.full():
                data_queue.put(latest_data)
            else:
                data_queue.get()  # Remove the oldest item if the queue is full
                data_queue.put(latest_data)

        if not data_queue.empty():
            current_data = data_queue.queue[-1]  # Always use the most recent data point
            display_data_graphics(current_data)
            time.sleep(15)  # Display each data point for 15 seconds

if __name__ == '__main__':
    main()
