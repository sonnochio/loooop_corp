import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
from firebase_admin import firestore, credentials
import firebase_admin
import time



# Initialize Cloud Firestore
cred = credentials.Certificate('fb_secret.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

# This function fetches the latest data point from 'data_points' collection
def fetch_latest_data_point(last_timestamp=None):
    query = db.collection('data_points').order_by('timestamp', direction=firestore.Query.DESCENDING)
    if last_timestamp:
        query = query.where('timestamp', '>', last_timestamp)
    docs = query.limit(1).stream()

    for doc in docs:
        return doc.to_dict()
    return None

# This function displays the graphics based on the fetched data
def display_data_graphics(data):
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

    plt.tight_layout()
    plt.show()

def main():
    last_timestamp = None

    while True:
        latest_data = fetch_latest_data_point(last_timestamp)
        if latest_data:
            display_data_graphics(latest_data)
            last_timestamp = latest_data['timestamp']
        time.sleep(15)  # Check for new data points every 15 seconds

if __name__ == '__main__':
    main()
