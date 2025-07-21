import pymongo
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bson.objectid import ObjectId

class DatabaseManager:
    def __init__(self, mongo_uri, db_name="cctv-db", collection_name="profiles"):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.feature_cache = self._load_features_to_cache()
        self.similarity_threshold = 0.8  # Tune this threshold

    def _load_features_to_cache(self):
        """Load all feature galleries into memory for faster matching."""
        cache = {}
        for doc in self.collection.find({}, {"_id": 1, "feature_gallery": 1}):
            cache[str(doc["_id"])] = np.array(doc["feature_gallery"])
        print(f"Loaded {len(cache)} profiles into feature cache.")
        return cache
    
    def match_or_create_person(self, new_feature, camera_id, bbox, suspicious_category="normal"):
        """
        Tries to match a feature vector to an existing person.
        If no match is found, creates a new person profile.
        Returns the global ID of the person.
        """
        new_feature_np = new_feature.reshape(1, -1)
        best_match_id = None
        max_similarity = -1

        if not self.feature_cache: # Handle case where cache is empty
            return self.create_new_person(new_feature, camera_id, bbox, suspicious_category)

        # Search for the best match in the cache
        for global_id, gallery in self.feature_cache.items():
            similarity = cosine_similarity(new_feature_np, gallery).max()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = global_id

        # Decision: Match or Create New
        if max_similarity > self.similarity_threshold:
            # Matched an existing person
            global_id = best_match_id
            self.update_person(global_id, new_feature, camera_id, bbox, suspicious_category)
        else:
            # No suitable match found, create a new person
            global_id = self.create_new_person(new_feature, camera_id, bbox, suspicious_category)

        return global_id
    
    def create_new_person(self, feature, camera_id, bbox, suspicious_category="normal"):
        """Creates a new document for a new person."""
        person_doc = {
            "source_id": None, # Can be updated later (e.g., employee ID)
            "status": "active",
            "first_seen_timestamp": datetime.utcnow(),
            "last_seen_timestamp": datetime.utcnow(),
            "last_seen_camera_id": camera_id,
            "feature_gallery": [feature.tolist()],
            "path_history": [{
                "ts": datetime.utcnow(),
                "cam": camera_id,
                "loc": [int(c) for c in bbox]
            }],
            "suspicious_category": suspicious_category  # 🆕 New field
        }
        result = self.collection.insert_one(person_doc)
        global_id = str(result.inserted_id)
        # Update cache
        self.feature_cache[global_id] = np.array([feature.tolist()])
        print(f"Created new person with global ID: {global_id}")
        return global_id
    
    def update_person(self, global_id, new_feature, camera_id, bbox, suspicious_category="normal"):
        """Updates an existing person's profile."""
        update_query = {
            "$set": {
                "last_seen_timestamp": datetime.utcnow(),
                "last_seen_camera_id": camera_id,
                "status": "active",
                "suspicious_category": suspicious_category  # 🆕 Update with latest category
            },
            "$push": {
                "feature_gallery": new_feature.tolist(),
                "path_history": {
                    "ts": datetime.utcnow(),
                    "cam": camera_id,
                    "loc": [int(c) for c in bbox]
                }
            }
        }
        self.collection.update_one({"_id": ObjectId(global_id)}, update_query)
        # Update cache
        self.feature_cache[global_id] = np.vstack(
            [self.feature_cache[global_id], new_feature]
        )