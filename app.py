"""
Flask Web Application for AI Teaching Assistant
Provides comprehensive REST APIs for chat, multimodal material ingestion,
interactive quizzes, executive study summaries, and learning analytics.
"""
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from teaching_assistant import AITeachingAssistant
from user_manager import UserManager
from firestore_manager import firestore_manager

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
CORS(app)

# Initialize Core Teaching Assistant and User Management
ta = AITeachingAssistant()
user_manager = UserManager()


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Render the Modern Web App UI"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check and configuration status"""
    stats = ta.vector_store_manager.get_stats()
    return jsonify({
        'status': 'healthy',
        'api_key_configured': bool(ta.api_key),
        'active_model': ta.model_name,
        'available_models': Config.AVAILABLE_MODELS,
        'total_documents': stats.get('total_documents', 0),
        'total_chunks': stats.get('total_chunks', 0)
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update API key or active Gemini model"""
    try:
        data = request.get_json() or {}
        if 'api_key' in data:
            new_key = data['api_key'].strip()
            ta.set_api_key(new_key)

        if 'model' in data:
            new_model = data['model'].strip()
            if new_model in Config.AVAILABLE_MODELS:
                ta.set_model(new_model)

        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'api_key_configured': bool(ta.api_key),
            'active_model': ta.model_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/auth/config', methods=['GET'])
def auth_config():
    """Expose public authentication configuration (Google Client ID)"""
    return jsonify({
        'google_client_id': Config.GOOGLE_CLIENT_ID or '',
        'google_auth_enabled': bool(Config.GOOGLE_CLIENT_ID)
    })


@app.route('/api/auth/google', methods=['POST'])
def auth_google():
    """Verify Google OAuth2 ID Token and authenticate student"""
    try:
        data = request.get_json() or {}
        credential = data.get('credential')
        if not credential:
            return jsonify({'success': False, 'error': 'Missing Google authentication credential token.'}), 400

        client_id = Config.GOOGLE_CLIENT_ID or None
        id_info = None

        # Method 1: Google oauth2 id_token verifier
        try:
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests
            id_info = id_token.verify_oauth2_token(
                credential,
                google_requests.Request(),
                client_id
            )
        except Exception as verify_err:
            # Method 2: Fallback to Google TokenInfo REST API
            try:
                import requests as http_requests
                tokeninfo_resp = http_requests.get(
                    f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}",
                    timeout=5
                )
                if tokeninfo_resp.status_code == 200:
                    id_info = tokeninfo_resp.json()
                else:
                    return jsonify({'success': False, 'error': f'Google token verification error: {str(verify_err)}'}), 401
            except Exception as http_err:
                return jsonify({'success': False, 'error': f'Verification server unreachable: {str(http_err)}'}), 500

        if not id_info or 'email' not in id_info:
            return jsonify({'success': False, 'error': 'Invalid Google account payload received.'}), 401

        email = id_info.get('email', '')
        name = id_info.get('name') or id_info.get('given_name') or email.split('@')[0]
        picture = id_info.get('picture', '')
        google_sub = id_info.get('sub', '')

        user, err = user_manager.authenticate_or_create_google_user(
            email=email,
            name=name,
            picture=picture,
            google_sub=google_sub
        )
        if err:
            return jsonify({'success': False, 'error': err}), 400

        return jsonify({
            'success': True,
            'user': user,
            'message': f'Signed in successfully with Google as {user.get("name")}!'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auth/profile', methods=['GET', 'POST'])
def auth_profile():
    """Get or update student personalization profile"""
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            email = request.args.get('email')
            if not user_id and not email:
                return jsonify({'success': True, 'user': None})

            user = user_manager.get_user_by_id(user_id) if user_id else None
            if not user:
                return jsonify({'error': 'User not found'}), 404
            return jsonify({'success': True, 'user': user})
        else:
            data = request.get_json() or {}
            user_id = data.get('user_id')
            updates = data.get('updates', {})
            user, err = user_manager.update_profile(user_id, updates)
            if err:
                return jsonify({'success': False, 'error': err}), 400
            return jsonify({'success': True, 'user': user, 'message': 'Profile updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Ask a question to Professor Nova (RAG Powered & Personalized)"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question in request body'}), 400

        question = data['question']
        chat_history = data.get('history', [])
        student_profile = data.get('profile')
        user_id = data.get('user_id')

        if not student_profile and user_id:
            student_profile = user_manager.get_user_by_id(user_id)

        response = ta.ask(question, chat_history=chat_history, student_profile=student_profile)

        # Persist doubt Q&A to Cloud Firestore for this student
        if user_id and response and response.get('answer'):
            try:
                firestore_manager.save_chat_message(
                    user_id=user_id,
                    question=question,
                    answer=response.get('answer', ''),
                    sources=response.get('sources', [])
                )
            except Exception as fe:
                app.logger.warning(f"Failed to persist chat to Firestore: {fe}")

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/history', methods=['GET'])
def get_student_chat_history():
    """Retrieve persistent doubt inquiry history from Cloud Firestore"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'history': []})
    history = firestore_manager.get_chat_history(user_id)
    return jsonify({'success': True, 'history': history})


@app.route('/api/chat/clear', methods=['POST'])
def clear_student_chat_history():
    """Clear doubt inquiries in Cloud Firestore for student"""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    if user_id:
        firestore_manager.clear_chat_history(user_id)
    return jsonify({'success': True})


@app.route('/api/quiz/save', methods=['POST'])
def save_student_quiz():
    """Persist student quiz performance to Cloud Firestore"""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    quiz_data = data.get('quiz', {})
    if user_id and quiz_data:
        firestore_manager.save_quiz_result(user_id, quiz_data)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Missing user_id or quiz data'}), 400


@app.route('/api/quiz/history', methods=['GET'])
def get_student_quiz_history():
    """Retrieve completed quizzes for student from Cloud Firestore"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'quizzes': []})
    quizzes = firestore_manager.get_quiz_history(user_id)
    return jsonify({'success': True, 'quizzes': quizzes})


@app.route('/api/upload-files', methods=['POST'])
def upload_files():
    """Upload and index multiple course files (PDF, DOCX, PPTX, TXT)"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        saved_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                file.save(save_path)
                saved_paths.append(save_path)

        if not saved_paths:
            return jsonify({'error': 'No valid files uploaded. Supported: PDF, DOCX, PPTX, TXT, MD'}), 400

        result = ta.ingest_files(saved_paths)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/youtube', methods=['POST'])
def ingest_youtube():
    """Transcribe and index YouTube video lecture URL"""
    try:
        data = request.get_json() or {}
        url = data.get('url', '').strip()
        if not url:
            return jsonify({'error': 'Missing YouTube video URL'}), 400

        result = ta.ingest_youtube(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/wikipedia', methods=['POST'])
def ingest_wikipedia():
    """Fetch and index Wikipedia academic topic"""
    try:
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Missing Wikipedia topic query'}), 400

        result = ta.ingest_wikipedia(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/direct-note', methods=['POST'])
def ingest_direct_note():
    """Index direct lecture note text"""
    try:
        data = request.get_json() or {}
        title = data.get('title', 'Lecture Note').strip()
        content = data.get('content', '').strip()

        if not content:
            return jsonify({'error': 'Note content cannot be empty'}), 400

        result = ta.ingest_text_note(title, content)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/materials', methods=['GET', 'DELETE'])
def manage_materials():
    """Inspect or delete indexed course materials"""
    if request.method == 'GET':
        stats = ta.vector_store_manager.get_stats()
        return jsonify(stats)
    elif request.method == 'DELETE':
        data = request.get_json() or {}
        source = data.get('source')
        if source:
            deleted = ta.vector_store_manager.delete_source(source)
            return jsonify({'message': f'Deleted {deleted} chunks for source: {source}'})
        else:
            ta.vector_store_manager.clear()
            return jsonify({'message': 'Knowledge base cleared successfully'})


@app.route('/api/quiz/generate', methods=['POST'])
def generate_quiz():
    """Generate an interactive practice quiz from course notes"""
    try:
        data = request.get_json() or {}
        topic = data.get('topic')
        num_q = int(data.get('num_questions', 5))
        difficulty = data.get('difficulty', 'Medium')

        result = ta.generate_quiz(topic=topic, num_questions=num_q, difficulty=difficulty)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz():
    """Submit quiz results and update student progress analytics"""
    try:
        data = request.get_json() or {}
        ta.analytics.log_quiz_result(data)
        return jsonify({'success': True, 'message': 'Quiz progress recorded'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary/generate', methods=['POST'])
def generate_summary():
    """Generate executive summary and flashcards"""
    try:
        data = request.get_json() or {}
        topic = data.get('topic')
        result = ta.generate_summary(topic=topic)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get student learning dashboard stats and past doubts"""
    try:
        dashboard = ta.analytics.get_dashboard_data()
        return jsonify(dashboard)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET', 'DELETE'])
def manage_history():
    """Get or clear recorded doubt history"""
    try:
        if request.method == 'DELETE':
            success = ta.analytics.clear_history()
            return jsonify({'success': success, 'message': 'Doubt history cleared'})
        else:
            dashboard = ta.analytics.get_dashboard_data()
            return jsonify({
                'history': dashboard.get('recent_history', []),
                'stats': {
                    'total_queries': dashboard.get('total_queries', 0),
                    'total_quizzes': dashboard.get('total_quizzes_taken', 0),
                    'accuracy': dashboard.get('accuracy_percent', 0)
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 65)
    print("   AI TEACHING ASSISTANT - FULL-STACK PLATFORM")
    print("=" * 65)
    print(f" Web UI: http://localhost:{port}")
    print(f" LLM:    Google Gemini ({ta.model_name})")
    print(f" Status: {'API Key Active' if ta.api_key else 'Setup Mode (Set key in Settings tab)'}")
    print("=" * 65 + "\n")

    app.run(debug=True, host='0.0.0.0', port=port)
