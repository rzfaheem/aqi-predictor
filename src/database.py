"""
Database Module - MongoDB connection and CRUD operations.
"""

from pymongo import MongoClient
from datetime import datetime
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


class Database:
    """Handles all MongoDB operations for data storage and retrieval."""
    
    def __init__(self):
        import ssl
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        self.client = MongoClient(
            config.MONGODB_CONNECTION_STRING,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
            serverSelectionTimeoutMS=10000
        )
        
        self.db = self.client[config.MONGODB_DATABASE_NAME]
        self.raw_data = self.db[config.COLLECTION_RAW_DATA]
        self.features = self.db[config.COLLECTION_FEATURES]
        self.models = self.db[config.COLLECTION_MODELS]
        
        print(f"✅ Connected to MongoDB database: {config.MONGODB_DATABASE_NAME}")
    
    # --- Raw Data Operations ---
    
    def save_raw_data(self, data: dict) -> str:
        """Save raw weather/pollution data to database."""
        data["saved_at"] = datetime.utcnow()
        result = self.raw_data.insert_one(data)
        print(f"✅ Saved raw data with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def get_raw_data(self, start_date: datetime = None, end_date: datetime = None) -> list:
        """Retrieve raw data, optionally filtered by date range."""
        query = {}
        
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        cursor = self.raw_data.find(query).sort("timestamp", 1)
        return list(cursor)
    
    # --- Feature Store Operations ---
    
    def save_features(self, features: dict) -> str:
        """Save processed features to the feature store."""
        features["saved_at"] = datetime.utcnow()
        result = self.features.insert_one(features)
        print(f"✅ Saved features with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def save_features_batch(self, features_list: list) -> int:
        """Save multiple feature records at once."""
        if not features_list:
            return 0
        
        for f in features_list:
            f["saved_at"] = datetime.utcnow()
        
        result = self.features.insert_many(features_list)
        count = len(result.inserted_ids)
        print(f"✅ Saved {count} feature records")
        return count
    
    def get_features(self, start_date: datetime = None, end_date: datetime = None) -> list:
        """Retrieve features, optionally filtered by date range."""
        query = {}
        
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        cursor = self.features.find(query).sort("timestamp", 1)
        return list(cursor)
    
    def get_latest_features(self, n: int = 1) -> list:
        """Get the most recent n feature records."""
        cursor = self.features.find().sort("timestamp", -1).limit(n)
        return list(cursor)
    
    # --- Model Registry Operations ---
    
    def save_model_info(self, model_info: dict) -> str:
        """Save trained model metadata to the registry."""
        model_info["saved_at"] = datetime.utcnow()
        result = self.models.insert_one(model_info)
        print(f"✅ Saved model info with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def get_best_model(self) -> dict:
        """Get the best model by lowest RMSE."""
        cursor = self.models.find().sort("metrics.rmse", 1).limit(1)
        models = list(cursor)
        return models[0] if models else None
    
    def get_latest_model(self) -> dict:
        """Get the most recently trained model."""
        cursor = self.models.find().sort("trained_at", -1).limit(1)
        models = list(cursor)
        return models[0] if models else None
    
    def save_model_binary(self, model_data: bytes, model_name: str, metrics: dict, feature_names: list) -> str:
        """Save model binary to MongoDB for cloud deployment."""
        import bson
        
        model_storage = self.db["model_storage"]
        model_storage.delete_many({})  # Keep only latest
        
        doc = {
            "model_name": model_name,
            "model_binary": bson.Binary(model_data),
            "metrics": metrics,
            "feature_names": feature_names,
            "saved_at": datetime.utcnow()
        }
        result = model_storage.insert_one(doc)
        print(f"✅ Saved model binary to MongoDB (ID: {result.inserted_id})")
        return str(result.inserted_id)
    
    def load_model_binary(self) -> dict:
        """Load model binary from MongoDB."""
        model_storage = self.db["model_storage"]
        model_doc = model_storage.find_one({}, sort=[("saved_at", -1)])
        
        if model_doc:
            return {
                "model_binary": model_doc["model_binary"],
                "model_name": model_doc.get("model_name", "Unknown"),
                "metrics": model_doc.get("metrics", {}),
                "feature_names": model_doc.get("feature_names", [])
            }
        return None
    
    # --- Utility Functions ---
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            self.client.admin.command('ping')
            print("✅ MongoDB connection is working!")
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
    
    def get_collection_stats(self) -> dict:
        """Get document counts for each collection."""
        return {
            "raw_data_count": self.raw_data.count_documents({}),
            "features_count": self.features.count_documents({}),
            "models_count": self.models.count_documents({})
        }
    
    def clear_collection(self, collection_name: str) -> int:
        """Clear all documents from a collection."""
        collection_map = {
            "raw_data": self.raw_data,
            "features": self.features,
            "models": self.models
        }
        
        if collection_name not in collection_map:
            print(f"❌ Unknown collection: {collection_name}")
            return 0
        
        result = collection_map[collection_name].delete_many({})
        print(f"🗑️ Deleted {result.deleted_count} documents from {collection_name}")
        return result.deleted_count


if __name__ == "__main__":
    db = Database()
    db.test_connection()
    
    stats = db.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Raw data: {stats['raw_data_count']}")
    print(f"   Features: {stats['features_count']}")
    print(f"   Models: {stats['models_count']}")
