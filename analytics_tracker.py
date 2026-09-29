"""
Student Analytics and Learning Progress Tracker
Persists doubt history, quiz results, and mastery metrics to disk
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from config import Config


class AnalyticsTracker:
    """Tracks and computes metrics on student learning activity"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or Config.ANALYTICS_PATH
        self.stats_file = os.path.join(self.data_dir, "student_stats.json")
        self.history_file = os.path.join(self.data_dir, "doubt_history.json")
        self.quiz_file = os.path.join(self.data_dir, "quiz_history.json")
        self._ensure_files()

    def _ensure_files(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.stats_file):
            default_stats = {
                "total_queries": 0,
                "total_quizzes_taken": 0,
                "total_questions_attempted": 0,
                "total_correct_answers": 0,
                "topics_explored": {},
                "study_streak_days": 1,
                "last_active": datetime.now().isoformat()
            }
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(default_stats, f, indent=2)

        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)

        if not os.path.exists(self.quiz_file):
            with open(self.quiz_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)

    def log_query(self, question: str, answer_snippet: str, sources: List[str]):
        """Record student doubt query in history and update metrics"""
        try:
            # Update stats
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            stats["total_queries"] = stats.get("total_queries", 0) + 1
            stats["last_active"] = datetime.now().isoformat()

            # Track topics or keywords
            for src in sources:
                stats["topics_explored"][src] = stats["topics_explored"].get(src, 0) + 1

            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            # Update history
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            entry = {
                "id": len(history) + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "question": question,
                "answer_preview": answer_snippet[:180] + "..." if len(answer_snippet) > 180 else answer_snippet,
                "sources": sources
            }
            history.insert(0, entry)  # Prepend newest

            # Keep latest 100 queries
            history = history[:100]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"Error logging query analytics: {e}")

    def log_quiz_result(self, quiz_data: Dict[str, Any]):
        """Record quiz performance results"""
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            attempted = quiz_data.get("total", 0)
            correct = quiz_data.get("score", 0)

            stats["total_quizzes_taken"] = stats.get("total_quizzes_taken", 0) + 1
            stats["total_questions_attempted"] = stats.get("total_questions_attempted", 0) + attempted
            stats["total_correct_answers"] = stats.get("total_correct_answers", 0) + correct

            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            with open(self.quiz_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            quiz_entry = {
                "id": len(history) + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "score": correct,
                "total": attempted,
                "accuracy": round((correct / max(attempted, 1)) * 100, 1),
                "topic": quiz_data.get("topic", "General Course Review")
            }
            history.insert(0, quiz_entry)
            history = history[:50]

            with open(self.quiz_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"Error logging quiz analytics: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Aggregate stats and recent activity for the UI dashboard"""
        self._ensure_files()
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            with open(self.quiz_file, 'r', encoding='utf-8') as f:
                quizzes = json.load(f)

            attempted = stats.get("total_questions_attempted", 0)
            correct = stats.get("total_correct_answers", 0)
            accuracy = round((correct / max(attempted, 1)) * 100, 1) if attempted > 0 else 0

            return {
                "total_queries": stats.get("total_queries", 0),
                "total_quizzes_taken": stats.get("total_quizzes_taken", 0),
                "accuracy_percent": accuracy,
                "study_streak_days": stats.get("study_streak_days", 1),
                "topics_explored": stats.get("topics_explored", {}),
                "recent_history": history[:15],
                "recent_quizzes": quizzes[:10]
            }
        except Exception as e:
            print(f"Error fetching dashboard data: {e}")
            return {
                "total_queries": 0,
                "total_quizzes_taken": 0,
                "accuracy_percent": 0,
                "study_streak_days": 1,
                "topics_explored": {},
                "recent_history": [],
                "recent_quizzes": []
            }

    def clear_history(self) -> bool:
        """Reset student doubt history"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            return True
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False
