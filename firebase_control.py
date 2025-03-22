
import firebase_admin
from firebase_admin import credentials, firestore, storage
import time


def connect_to_firebase(json_path):
    """اتصال بـ Firebase باستخدام ملف JSON."""
    print(f"Attempting to connect to Firebase using path: {json_path}")
    for app in firebase_admin._apps.values():
        print(f"Deleting existing Firebase app: {app.name}")
        firebase_admin.delete_app(app)
    cred = credentials.Certificate(json_path)
    print("Initializing Firebase app with storage")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'petsapp-fad2e.appspot.com'  # Replace with your bucket URL
    })
    return firestore.client()


def get_all_credentials(db, collection_name):
    """جلب جميع الأسماء وكلمات المرور ومعرفات المستندات من مجموعة معينة."""
    print(f"Retrieving credentials from collection: {collection_name}")
    docs = db.collection(collection_name).stream()
    names = []
    passwords = []
    ids = []
    for doc in docs:
        print(f"Processing document ID: {doc.id}")
        doc_data = doc.to_dict()
        names.append(doc_data.get("name", "No Name"))
        passwords.append(doc_data.get("password", "No Password"))
        ids.append(doc.id)
    print(f"Names retrieved: {names}")
    print(f"Passwords retrieved: {passwords}")
    print(f"IDs retrieved: {ids}")
    return names, passwords, ids


def upload_file(db, local_path, remote_path):
    """Upload a file to Firebase Storage and return the download URL."""
    print(f"Uploading file from {local_path} to {remote_path}")
    try:
        bucket = storage.bucket()
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        blob.make_public()
        url = blob.public_url
        print(f"File uploaded successfully. URL: {url}")
        return url
    except Exception as e:
        print(f"Error uploading file to Firebase: {e}")
        return None


def store_detection(db, data):
    """Store detection data in Firestore under 'detections' collection."""
    try:
        # Add to 'detections' collection with auto-generated ID
        detection_ref = db.collection("detections").document()
        detection_ref.set({
            "date": data["date"],  # Keeps the date as a string like "2025-02-19 00:06:55.516100"
            "discomfortReason": None,  # null value
            "employeeId": None,  # null value
            "id": detection_ref.id,  # Uses the auto-generated Firestore document ID
            "notes": None,  # null value
            "restaurantId": data["restaurant_id"],
            "screenShoot": data["screenshot_url"],
            "tableNumber": data["table_number"],
            "timestamp": firestore.SERVER_TIMESTAMP  # Add server timestamp
        })

        print(f"Detection stored in Firestore with ID: {detection_ref.id}")
        return detection_ref.id
    except Exception as e:
        print(f"Error storing detection in Firestore: {e}")
        return None

if __name__ == "__main__":
    db = connect_to_firebase('json_file.json')
    names, passwords, ids = get_all_credentials(db, "restaurants")
    print(f"Names: {names}")
    print(f"Passwords: {passwords}")
    print(f"IDs: {ids}")

    test_file = "test.jpg"
    if os.path.exists(test_file):
        url = upload_file(db, test_file, f"test/{os.path.basename(test_file)}")
        print(f"Test file uploaded to: {url}")
    else:
        print(f"Test file {test_file} not found")