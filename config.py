"""
Configuration settings for AI Teaching Assistant
Supports Google Gemini, Local Storage, RAG parameters, and File Uploads
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Google Authentication (OAuth 2.0)
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
    GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')

    # Google Gemini Settings
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    DEFAULT_MODEL = os.getenv('GEMINI_MODEL', 'gemini-flash-lite-latest')
    AVAILABLE_MODELS = [
        'gemini-flash-lite-latest',
        'gemini-flash-latest',
        'gemini-2.5-flash',
        'gemini-2.5-pro'
    ]
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-2')
    
    # Text Splitting Settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 800))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 150))
    
    # Storage Paths (use /tmp on Vercel serverless)
    IS_VERCEL = os.getenv('VERCEL') == '1'
    STORAGE_ROOT = '/tmp' if IS_VERCEL else BASE_DIR

    UPLOAD_FOLDER = os.path.join(STORAGE_ROOT, 'uploads')
    VECTOR_STORE_PATH = os.path.join(STORAGE_ROOT, 'vector_store_data')
    ANALYTICS_PATH = os.path.join(STORAGE_ROOT, 'analytics_data')
    
    # RAG Retrieval Settings
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 6))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))
    
    # Supported File Extensions
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx', 'txt', 'md', 'csv', 'json', 'py', 'java', 'c', 'cpp'}
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max file size

    @classmethod
    def update_api_key(cls, new_key: str):
        """Update API key at runtime and persist to .env if not on Vercel"""
        cls.GEMINI_API_KEY = new_key.strip()
        if cls.IS_VERCEL:
            return

        env_file = os.path.join(BASE_DIR, '.env')
        try:
            lines = []
            if os.path.exists(env_file):
                with open(env_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            key_found = False
            new_lines = []
            for line in lines:
                if line.startswith('GEMINI_API_KEY='):
                    new_lines.append(f"GEMINI_API_KEY={cls.GEMINI_API_KEY}\n")
                    key_found = True
                else:
                    new_lines.append(line)
            
            if not key_found:
                new_lines.append(f"GEMINI_API_KEY={cls.GEMINI_API_KEY}\n")
                
            with open(env_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"Notice: Could not write .env file: {e}")

# Ensure directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(Config.ANALYTICS_PATH, exist_ok=True)
