"""
Database Module - MongoDB Connection and Operations
====================================================

WHAT THIS FILE DOES:
- Connects to MongoDB Atlas (our cloud database)
- Provides functions to save and retrieve data
- Works like a helper to talk to our database

WHAT IS MONGODB?
- MongoDB is a "NoSQL" database (stores data as documents, like JSON)
- Unlike SQL databases (rows and columns), MongoDB stores flexible documents
- Each document is like a Python dictionary: {"name": "John", "age": 25}

COLLECTIONS (like tables in SQL):
- raw_weather_data: Stores the data we fetch from APIs
- features: Stores processed features for model training
- model_registry: Stores information about trained models
"""

from pymongo import MongoClient
from datetime import datetime
import sys
import os

# Add project root directory to path so we can import config
# This works both locally and on GitHub Actions
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


class Database:
    """
    A class to handle all MongoDB operations.
    
    WHY USE A CLASS?
    - Keeps all database-related code organized in one place
    - The connection is created once and reused (efficient)
    - Easy to use: just call db.save_data(...) or db.get_data(...)
    """
    
    def __init__(self):
        """
        Initialize the database connection.
        
        This runs when you create a Database object:
            db = Database()  # This line calls __init__
        """
        import ssl
        
        # Create a custom SSL context that doesn't verify certificates
        # This fixes connection issues on Windows
        # Note: This is fine for learning/development, but production should use proper certs
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Connect to MongoDB Atlas using our connection string from config
        self.client = MongoClient(
            config.MONGODB_CONNECTION_STRING,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
            serverSelectionTimeoutMS=10000  # 10 second timeout
        )
        
        # Select our database (like choosing a folder)
        self.db = self.client[config.MONGODB_DATABASE_NAME]
        
        # Select our collections (like choosing files within the folder)
        self.raw_data = self.db[config.COLLECTION_RAW_DATA]
        self.features = self.db[config.COLLECTION_FEATURES]
        self.models = self.db[config.COLLECTION_MODELS]
        
        print(f"✅ Connected to MongoDB database: {config.MONGODB_DATABASE_NAME}")
    
    # ========================================
    # RAW DATA OPERATIONS
    # ========================================
    
    def save_raw_data(self, data: dict) -> str:
        """
        Save raw weather/pollution data from API to database.
        
        Parameters:
            data (dict): The data to save (from API response)
        
        Returns:
            str: The ID of the saved document
        
        Example:
            db.save_raw_data({"temperature": 25, "aqi": 150, "timestamp": "2024-01-23"})
        """
        # Add a timestamp for when we saved this data
        data["saved_at"] = datetime.utcnow()
        
        # Insert into MongoDB and get the generated ID
        result = self.raw_data.insert_one(data)
        
        print(f"✅ Saved raw data with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def get_raw_data(self, start_date: datetime = None, end_date: datetime = None) -> list:
        """
        Retrieve raw data from database.
        
        Parameters:
            start_date: Get data from this date (optional)
            end_date: Get data until this date (optional)
        
        Returns:
            list: List of data documents
        
        Example:
            # Get all data
            all_data = db.get_raw_data()
            
            # Get data for a specific date range
            from datetime import datetime
            data = db.get_raw_data(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31)
            )
        """
        # Build the query (filter)
        query = {}
        
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date  # $gte = greater than or equal
            if end_date:
                query["timestamp"]["$lte"] = end_date    # $lte = less than or equal
        
        # Find all documents matching the query
        # Sort by timestamp (oldest first)
        cursor = self.raw_data.find(query).sort("timestamp", 1)
        
        return list(cursor)
    
    # ========================================
    # FEATURES OPERATIONS (Feature Store)
    # ========================================
    
    def save_features(self, features: dict) -> str:
        """
        Save processed features to the Feature Store.
        
        Parameters:
            features (dict): The processed features to save
        
        Returns:
            str: The ID of the saved document
        """
        features["saved_at"] = datetime.utcnow()
        result = self.features.insert_one(features)
        print(f"✅ Saved features with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def save_features_batch(self, features_list: list) -> int:
        """
        Save multiple feature records at once (for historical data).
        
        Parameters:
            features_list (list): List of feature dictionaries
        
        Returns:
            int: Number of documents saved
        """
        if not features_list:
            return 0
        
        # Add saved_at timestamp to each
        for f in features_list:
            f["saved_at"] = datetime.utcnow()
        
        result = self.features.insert_many(features_list)
        count = len(result.inserted_ids)
        print(f"✅ Saved {count} feature records")
        return count
    
    def get_features(self, start_date: datetime = None, end_date: datetime = None) -> list:
        """
        Retrieve features from the Feature Store.
        
        Parameters:
            start_date: Get features from this date (optional)
            end_date: Get features until this date (optional)
        
        Returns:
            list: List of feature documents
        """
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
        """
        Get the most recent n feature records.
        
        Parameters:
            n (int): Number of recent records to get
        
        Returns:
            list: List of the n most recent feature documents
        """
        cursor = self.features.find().sort("timestamp", -1).limit(n)
        return list(cursor)
    
    # ========================================
    # MODEL REGISTRY OPERATIONS
    # ========================================
    
    def save_model_info(self, model_info: dict) -> str:
        """
        Save trained model information to the Model Registry.
        
        Parameters:
            model_info (dict): Information about the trained model including:
                - model_name: Name of the model (e.g., "random_forest")
                - model_path: Path where model file is saved
                - metrics: Dictionary of evaluation metrics (RMSE, MAE, R2)
                - trained_at: When the model was trained
                - features_used: List of feature names used
        
        Returns:
            str: The ID of the saved document
        """
        model_info["saved_at"] = datetime.utcnow()
        result = self.models.insert_one(model_info)
        print(f"✅ Saved model info with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def get_best_model(self) -> dict:
        """
        Get the best performing model from the registry.
        We determine "best" by lowest RMSE (Root Mean Square Error).
        
        Returns:
            dict: Information about the best model, or None if no models exist
        """
        # Sort by RMSE (ascending) and get the first one
        cursor = self.models.find().sort("metrics.rmse", 1).limit(1)
        models = list(cursor)
        
        if models:
            return models[0]
        return None
    
    def get_latest_model(self) -> dict:
        """
        Get the most recently trained model.
        
        Returns:
            dict: Information about the latest model, or None if no models exist
        """
        cursor = self.models.find().sort("trained_at", -1).limit(1)
        models = list(cursor)
        
        if models:
            return models[0]
        return None
    
    def save_model_binary(self, model_data: bytes, model_name: str, metrics: dict, feature_names: list) -> str:
        """
        Save the actual model binary to MongoDB for cloud deployment.
        
        Parameters:
            model_data (bytes): Pickled model binary data
            model_name (str): Name of the model
            metrics (dict): Model performance metrics
            feature_names (list): List of feature names used
        
        Returns:
            str: The ID of the saved document
        """
        import bson
        
        # Create model storage collection if not exists
        model_storage = self.db["model_storage"]
        
        # Remove old models (keep only latest)
        model_storage.delete_many({})
        
        # Save new model
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
        """
        Load the model binary from MongoDB.
        
        Returns:
            dict: Model data including binary, or None if not found
        """
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
    
    # ========================================
    # UTILITY FUNCTIONS
    # ========================================
    
    def test_connection(self) -> bool:
        """
        Test if the database connection is working.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # The ping command is used to test connection
            self.client.admin.command('ping')
            print("✅ MongoDB connection is working!")
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about our collections.
        
        Returns:
            dict: Count of documents in each collection
        """
        stats = {
            "raw_data_count": self.raw_data.count_documents({}),
            "features_count": self.features.count_documents({}),
            "models_count": self.models.count_documents({})
        }
        return stats
    
    def clear_collection(self, collection_name: str) -> int:
        """
        Clear all documents from a collection (use with caution!).
        
        Parameters:
            collection_name: One of "raw_data", "features", or "models"
        
        Returns:
            int: Number of documents deleted
        """
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


# ========================================
# QUICK TEST (runs when you execute this file directly)
# ========================================

if __name__ == "__main__":
    """
    This code only runs when you execute this file directly:
        python src/database.py
    
    It won't run when you import this module in other files.
    """
    print("Testing database connection...")
    print("=" * 50)
    
    # Create database instance
    db = Database()
    
    # Test connection
    db.test_connection()
    
    # Show current stats
    stats = db.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Raw data documents: {stats['raw_data_count']}")
    print(f"   Features documents: {stats['features_count']}")
    print(f"   Models documents: {stats['models_count']}")
