"""
User Management and Authentication System
Provides secure password hashing, user registration, profile settings,
and per-user data isolation for personalized tutoring.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config


class UserManager:
    """Manages student user accounts, authentication, and personalized profiles"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or Config.STORAGE_ROOT
        self.users_file = os.path.join(self.data_dir, "users_db.json")
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.users_file):
            # Seed with a default student account for Abhinandan
            default_users = {}
            default_id = "user_" + uuid.uuid4().hex[:8]
            default_users[default_id] = {
                "id": default_id,
                "email": "student@university.edu",
                "name": "Abhinandan Dubey",
                "password_hash": generate_password_hash("password123"),
                "major": "Computer Science & Engineering",
                "academic_level": "Undergraduate (3rd Year)",
                "learning_style": "Intuitive Analogies & Practical Examples",
                "goal": "Semester Exam Prep & Deep Conceptual Mastery",
                "created_at": datetime.now().isoformat(),
                "last_login": datetime.now().isoformat()
            }
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, indent=2)

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_storage()
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_users(self, users: Dict[str, Dict[str, Any]]):
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2)

    def register(
        self,
        name: str,
        email: str,
        password: str,
        major: str = "Computer Science",
        academic_level: str = "Undergraduate",
        learning_style: str = "Intuitive Analogies & Code",
        goal: str = "Exam Preparation"
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Register a new student account"""
        name = name.strip()
        email = email.strip().lower()
        password = password.strip()

        if not name or not email or not password:
            return None, "Name, email, and password are required."
        if len(password) < 4:
            return None, "Password must be at least 4 characters long."

        users = self._load_users()
        for u in users.values():
            if u.get("email") == email:
                return None, "An account with this email already exists. Please sign in."

        user_id = "user_" + uuid.uuid4().hex[:10]
        user_record = {
            "id": user_id,
            "email": email,
            "name": name,
            "password_hash": generate_password_hash(password),
            "major": major.strip() or "General Academic Studies",
            "academic_level": academic_level.strip() or "Undergraduate",
            "learning_style": learning_style.strip() or "Intuitive Analogies & Code",
            "goal": goal.strip() or "Exam Preparation",
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat()
        }

        users[user_id] = user_record
        self._save_users(users)

        # Create isolated per-user analytics directory
        user_dir = self.get_user_data_dir(user_id)
        os.makedirs(user_dir, exist_ok=True)

        return self._sanitize_user(user_record), None

    def authenticate(self, email: str, password: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Verify student credentials and log in"""
        email = email.strip().lower()
        password = password.strip()

        users = self._load_users()
        for u in users.values():
            if u.get("email") == email:
                if check_password_hash(u.get("password_hash", ""), password):
                    u["last_login"] = datetime.now().isoformat()
                    self._save_users(users)
                    return self._sanitize_user(u), None
                else:
                    return None, "Invalid password. Please check your credentials."

        return None, "No account found with this email. Please sign up."

    def authenticate_or_create_google_user(
        self,
        email: str,
        name: str,
        picture: Optional[str] = None,
        google_sub: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Log in or automatically provision a student account via Google OAuth"""
        email = email.strip().lower()
        name = name.strip() or "Student"

        users = self._load_users()
        # Find existing user by email
        for user_id, u in users.items():
            if u.get("email") == email:
                if picture:
                    u["avatar_url"] = picture
                if google_sub:
                    u["google_id"] = google_sub
                u["auth_provider"] = "google"
                u["last_login"] = datetime.now().isoformat()
                self._save_users(users)
                return self._sanitize_user(u), None

        # Provision a new account for this Google user
        user_id = "user_g_" + uuid.uuid4().hex[:10]
        user_record = {
            "id": user_id,
            "email": email,
            "name": name,
            "password_hash": "",
            "auth_provider": "google",
            "avatar_url": picture or "",
            "google_id": google_sub or "",
            "major": "Computer Science & Engineering",
            "academic_level": "Undergraduate",
            "learning_style": "Intuitive Analogies & Practical Examples",
            "goal": "Deep Conceptual Mastery & Course Success",
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat()
        }
        users[user_id] = user_record
        self._save_users(users)

        # Isolated directory for this student
        user_dir = self.get_user_data_dir(user_id)
        os.makedirs(user_dir, exist_ok=True)

        return self._sanitize_user(user_record), None


    def get_default_user(self) -> Optional[Dict[str, Any]]:
        """Retrieve the primary or first registered student profile"""
        users = self._load_users()
        if users:
            first_user = next(iter(users.values()))
            return self._sanitize_user(first_user)
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile by ID"""
        users = self._load_users()
        if user_id in users:
            return self._sanitize_user(users[user_id])
        return None

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Update student personalization details"""
        users = self._load_users()
        if user_id not in users:
            return None, "User not found."

        user = users[user_id]
        allowed_fields = ["name", "major", "academic_level", "learning_style", "goal"]
        for field in allowed_fields:
            if field in updates and updates[field]:
                user[field] = str(updates[field]).strip()

        if "password" in updates and len(updates["password"]) >= 4:
            user["password_hash"] = generate_password_hash(updates["password"])

        users[user_id] = user
        self._save_users(users)
        return self._sanitize_user(user), None

    def get_user_data_dir(self, user_id: str) -> str:
        """Returns the isolated storage directory for a specific student"""
        safe_id = "".join(c for c in user_id if c.isalnum() or c in "_-")
        return os.path.join(Config.STORAGE_ROOT, "users_data", safe_id)

    @staticmethod
    def _sanitize_user(user: Dict[str, Any]) -> Dict[str, Any]:
        """Strip password hash before returning user dictionary"""
        safe = dict(user)
        safe.pop("password_hash", None)
        return safe
