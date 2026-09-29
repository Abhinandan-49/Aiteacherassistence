"""
AI Teaching Assistant - Main Application Entrypoint
Provides both programmatic interface and CLI demonstration using Google Gemini & RAG
"""
import sys
from config import Config
from teaching_assistant import AITeachingAssistant

def main():
    print("=" * 65)
    print("  AI TEACHING ASSISTANT (Powered by Google Gemini & RAG)")
    print("=" * 65)
    
    ta = AITeachingAssistant()
    stats = ta.vector_store_manager.get_stats()
    print(f"Indexed Documents: {stats['total_documents']}")
    print(f"Total Chunks:      {stats['total_chunks']}")
    print(f"Gemini API Key:    {'Configured' if ta.api_key else 'Not configured (set in .env or Settings UI)'}")
    print(f"Active Model:      {ta.model_name}")
    print("-" * 65)
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nStudent Query: {query}\n")
        resp = ta.ask(query)
        print("Professor Nova's Response:\n")
        print(resp.get("answer", ""))
        print("\nFollow-up Questions:")
        for q in resp.get("follow_ups", []):
            print(f" - {q}")
    else:
        print("Tip: Run 'python app.py' to launch the interactive Full-Stack Web Application!")

if __name__ == "__main__":
    main()
