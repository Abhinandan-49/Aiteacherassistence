"""
Firestore Cloud Persistence Manager for Professor Nova AI Teaching Assistant
Integrates with Google Firebase Firestore to persist student profiles,
personalized learning preferences, doubt inquiry history, and quiz progress in the cloud.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from config import Config

logger = logging.getLogger(__name__)

# Lazy / safe import of Firebase Admin SDK
FIREBASE_AVAILABLE = False
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    logger.warning("firebase-admin is not installed. Cloud Firestore persistence will be disabled.")


class FirestoreManager:
    """Manages cloud document persistence in Google Cloud Firestore"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FirestoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self.db = None
        self.is_connected = False
        self._init_firestore()

    def _init_firestore(self):
        """Initialize Firebase Admin and Cloud Firestore client"""
        if not FIREBASE_AVAILABLE:
            logger.info("Firebase Admin SDK not available in environment.")
            return

        try:
            # Check if Firebase is already initialized in this process
            if not firebase_admin._apps:
                cred = None
                # Priority 1: Service Account JSON string in environment variable
                json_str = Config.FIREBASE_SERVICE_ACCOUNT_JSON or os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON', '')
                if json_str:
                    try:
                        cert_dict = json.loads(json_str)
                        cred = credentials.Certificate(cert_dict)
                        logger.info("Loaded Firebase credentials from environment JSON string.")
                    except Exception as e:
                        logger.warning(f"Could not parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}")

                # Priority 2: Service Account JSON file path
                if not cred and os.path.exists(Config.FIREBASE_CREDENTIALS_PATH):
                    try:
                        cred = credentials.Certificate(Config.FIREBASE_CREDENTIALS_PATH)
                        logger.info(f"Loaded Firebase credentials from file: {Config.FIREBASE_CREDENTIALS_PATH}")
                    except Exception as e:
                        logger.warning(f"Could not load credentials from file: {e}")

                # Priority 3: Fallback to application default credentials
                if not cred:
                    try:
                        cred = credentials.ApplicationDefault()
                    except Exception:
                        pass

                if cred:
                    firebase_admin.initialize_app(cred, {
                        'projectId': Config.FIREBASE_PROJECT_ID or 'ai-teacher-assistance-4381d'
                    })
                else:
                    logger.warning("No valid Firebase credentials found. Firestore disabled.")
                    return

            self.db = firestore.client()
            self.is_connected = True
            logger.info(f"Connected to Cloud Firestore (Project: {Config.FIREBASE_PROJECT_ID})")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            self.db = None
            self.is_connected = False

    # ================= USER PROFILES & SETTINGS =================

    def save_user(self, user_data: Dict[str, Any]) -> bool:
        """Store or update a user document in the 'users' collection"""
        if not self.is_connected or not self.db:
            return False

        user_id = user_data.get('id')
        if not user_id:
            return False

        try:
            doc_ref = self.db.collection('users').document(str(user_id))
            payload = dict(user_data)
            payload['updated_at'] = datetime.now().isoformat()
            doc_ref.set(payload, merge=True)
            return True
        except Exception as e:
            logger.error(f"Firestore save_user error: {e}")
            return False

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by unique ID"""
        if not self.is_connected or not self.db or not user_id:
            return None

        try:
            doc = self.db.collection('users').document(str(user_id)).get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Firestore get_user_by_id error: {e}")
        return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Query user by email"""
        if not self.is_connected or not self.db or not email:
            return None

        try:
            query = self.db.collection('users').where('email', '==', email.strip().lower()).limit(1).stream()
            for doc in query:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Firestore get_user_by_email error: {e}")
        return None

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Update student personalization details in Cloud Firestore"""
        if not self.is_connected or not self.db:
            return None, "Firestore not connected"

        try:
            doc_ref = self.db.collection('users').document(str(user_id))
            doc = doc_ref.get()
            if not doc.exists:
                return None, "User not found in Cloud Firestore."

            allowed_fields = ["name", "major", "academic_level", "learning_style", "goal"]
            clean_updates = {
                k: str(v).strip() for k, v in updates.items()
                if k in allowed_fields and v is not None
            }
            clean_updates['updated_at'] = datetime.now().isoformat()

            doc_ref.set(clean_updates, merge=True)
            updated_doc = doc_ref.get().to_dict()
            return updated_doc, None
        except Exception as e:
            logger.error(f"Firestore update_user_profile error: {e}")
            return None, str(e)

    # ================= CHAT SESSIONS & DOUBTS =================

    def save_chat_message(
        self,
        user_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Store an office hour doubt inquiry in Cloud Firestore"""
        if not self.is_connected or not self.db or not user_id:
            return False

        try:
            timestamp = datetime.now().isoformat()
            chat_record = {
                "question": question,
                "answer": answer,
                "sources": sources or [],
                "timestamp": timestamp,
                "created_at": firestore.SERVER_TIMESTAMP
            }
            # Store in subcollection users/{user_id}/chats
            self.db.collection('users').document(str(user_id)).collection('chats').add(chat_record)
            return True
        except Exception as e:
            logger.error(f"Firestore save_chat_message error: {e}")
            return False

    def get_chat_history(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetch past doubts and Q&A history for a student"""
        if not self.is_connected or not self.db or not user_id:
            return []

        try:
            chats_ref = (
                self.db.collection('users')
                .document(str(user_id))
                .collection('chats')
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            docs = chats_ref.stream()
            history = []
            for doc in docs:
                data = doc.to_dict()
                history.append({
                    "id": doc.id,
                    "question": data.get("question", ""),
                    "answer": data.get("answer", ""),
                    "sources": data.get("sources", []),
                    "timestamp": data.get("timestamp", "")
                })
            # Reverse so oldest in the batch is first for chronological display
            return list(reversed(history))
        except Exception as e:
            logger.error(f"Firestore get_chat_history error: {e}")
            return []

    def clear_chat_history(self, user_id: str) -> bool:
        """Clear chat history subcollection for a student"""
        if not self.is_connected or not self.db or not user_id:
            return False

        try:
            chats_ref = self.db.collection('users').document(str(user_id)).collection('chats')
            for doc in chats_ref.stream():
                doc.reference.delete()
            return True
        except Exception as e:
            logger.error(f"Firestore clear_chat_history error: {e}")
            return False

    # ================= QUIZZES & LEARNING PROGRESS =================

    def save_quiz_result(self, user_id: str, quiz_data: Dict[str, Any]) -> bool:
        """Save a quiz score and questions drill record"""
        if not self.is_connected or not self.db or not user_id:
            return False

        try:
            quiz_record = dict(quiz_data)
            quiz_record['timestamp'] = datetime.now().isoformat()
            self.db.collection('users').document(str(user_id)).collection('quizzes').add(quiz_record)
            return True
        except Exception as e:
            logger.error(f"Firestore save_quiz_result error: {e}")
            return False

    def get_quiz_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get past completed quizzes for a student"""
        if not self.is_connected or not self.db or not user_id:
            return []

        try:
            quizzes_ref = (
                self.db.collection('users')
                .document(str(user_id))
                .collection('quizzes')
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            return [doc.to_dict() for doc in quizzes_ref.stream()]
        except Exception as e:
            logger.error(f"Firestore get_quiz_history error: {e}")
            return []


# Global Firestore Manager Singleton
firestore_manager = FirestoreManager()
