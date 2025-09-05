import pandas as pd
from pymongo import MongoClient
import json

def upload_csv_to_mongodb(connection_string, db_name, collection_name, csv_file_path):
    """
    Connects to a MongoDB database, reads a CSV file, and uploads its
    full contents to a specified collection.
    """
    try:
        print("Connecting to local MongoDB...")
        client = MongoClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]
        print("Connection successful.")

        print(f"Reading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from the CSV file.")

        print("Converting data to JSON format...")
        data = json.loads(df.to_json(orient='records'))

        print(f"Uploading full dataset to the '{collection_name}' collection in the '{db_name}' database...")
        collection.delete_many({}) # Clear existing data in the collection
        collection.insert_many(data)
        print("Upload complete!")
        print(f"Total documents in collection: {collection.count_documents({})}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()
            print("Connection closed.")

if __name__ == '__main__':
    MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
    
    DB_NAME = "credit_risk_db"
    COLLECTION_NAME = "applicants"
    CSV_PATH = "Data/Raw_data/application_train.csv"
    
    upload_csv_to_mongodb(MONGO_CONNECTION_STRING, DB_NAME, COLLECTION_NAME, CSV_PATH)