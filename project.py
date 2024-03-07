import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
from firebase_admin import firestore, credentials
import firebase_admin
import time
import queue
import threading

# Initialize Cloud Firestore
cred = credentials.Certificate('fb_secret.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

# Initialize a queue to hold data points
data_queue = queue.Queue()

# This function fetches the latest data point from 'data_points' collection
def fetch_latest_data_point(last_timestamp=None):
    try:
        query = db.collection('data_points').order_by('timestamp', direction=firestore.Query.DESCENDING)
        if last_timestamp:
            query = query.where('timestamp', '>', last_timestamp)
        docs = query.limit(1).stream()

        for doc in docs:
            print("Fetched new document:", doc.to_dict())  # Debugging print
            return doc.to_dict()
    except Exception as e:
        print("Error fetching document:", e)
    return None

# Function to display data from the queue
def display_data_graphics(data):
    plt.ion()  # Interactive mode on
    plt.close('all')  # Closes all the figure windows before opening a new one

    response = requests.get(data['blob_url'])
    image = Image.open(io.BytesIO(response.content))
    rumors_text = "\n\n\n".join(data['rumors'])

    fig, axs = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1]})
    fig.patch.set_facecolor('black')

    axs[0].text(0, 1, rumors_text, ha='left', va='top', fontsize=20, color='white', 
                wrap=True, transform=axs[0].transAxes, family='monospace')
    axs[0].set_facecolor('black')
    axs[0].axis('off')

    axs[1].imshow(image)
    axs[1].axis('off')
    axs[1].set_facecolor('black')

    summarized_text = data['summarized_image_prompt']
    axs[1].text(0.5, 0, summarized_text, ha='center', va='top', fontsize=20, color='white', 
                wrap=True, transform=axs[1].transAxes, family='monospace')

    plt.show(block=False)  # Display the figure without blocking
    plt.pause(0.1)  # Short pause to ensure the plot updates

# Function to fetch data and add it to the queue
def fetch_and_enqueue_data():
    global last_timestamp
    while True:
        try:
            new_data = fetch_latest_data_point(last_timestamp)
            if new_data:
                print("Enqueued new data point.")
                data_queue.put(new_data)
                last_timestamp = new_data['timestamp']
            else:
                print("No new data to enqueue.")
        except Exception as e:
            print(f"Error during data fetching: {e}")
        time.sleep(0.5)  # Polling interval

# Function to display data from the queue
def display_data_from_queue():
    last_displayed_data = None
    while True:
        if not data_queue.empty():
            last_displayed_data = data_queue.get()
            print("Displaying new data point from queue.")
            display_data_graphics(last_displayed_data)
        elif last_displayed_data:
            # Re-display the last data point if the queue is empty
            print("Redisplaying last data point, queue is empty.")
            display_data_graphics(last_displayed_data)
        time.sleep(15)  # Check the queue every second

if __name__ == '__main__':
    last_timestamp = None  # Initialize last_timestamp outside the functions to be used globally

    # Start the fetch_and_enqueue_data function in a background thread
    fetch_thread = threading.Thread(target=fetch_and_enqueue_data, daemon=True)
    fetch_thread.start()

    # Run the display_data_from_queue function in the main thread
    display_data_from_queue()
