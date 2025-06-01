# db_utils.py
import sqlite3
import json
import os
import time
import uuid
import logging

logger = logging.getLogger(__name__)

DATABASE_NAME = "flashcard_db.sqlite"
MEDIA_DIR = "flashcard_media"
SETTINGS_FILE_PATH = "flash_app_settings.json" # Added here for consistency

# Field types configuration (can also live here or be passed around)
field_types_config = {
    "definition": {
        "label": "Definition(s)", "json_output": False, "allow_multiple": True,
        "prompt_template": "Provide a clear and concise definition for the English word/phrase: '{WORD_PHRASE}'. If multiple common meanings exist, provide the primary ones. Output only the definition text.",
        "user_input_label": "Your Definition"
    },
    "word_family": {
        "label": "Word Family", "json_output": True, "allow_multiple": False,
        "prompt_template": "Generate the derivational word family for the English word/phrase: '{WORD_PHRASE}'. Include related nouns, verbs, adjectives, and adverbs. Output as a JSON object: {\"root_word\": \"str\", \"nouns\": [\"str\"], \"verbs\": [\"str\"], \"adjectives\": [\"str\"], \"adverbs\": [\"str\"]}.",
        "user_input_label": "Your Word Family (JSON)",
        "template_for_user": json.dumps({"root_word": "", "nouns": [""], "verbs": [""], "adjectives": [""], "adverbs": [""]}, indent=4)
    },
    "example_sentence": {
        "label": "Example Sentence(s)", "json_output": False, "allow_multiple": True,
        "prompt_template": "Generate 1 distinct example sentence using the English word/phrase: '{WORD_PHRASE}'. Ensure it's suitable for an intermediate English learner. Output only the sentence.",
        "user_input_label": "Your Example Sentence"
    },
    "pairwise_dialogue": {
        "label": "Pairwise Dialogue (Q&A)", "json_output": True, "allow_multiple": True,
        "prompt_template": "Create a short dialogue (question & answer) for an English learner using the word/phrase: '{WORD_PHRASE}'. Output as JSON: {\"question\": \"str\", \"answer\": \"str\"}.",
        "user_input_label": "Your Q&A (JSON)",
        "template_for_user": json.dumps({"question": "", "answer": ""}, indent=4)
    }
}


def init_flashcard_db():
    if not os.path.exists(MEDIA_DIR):
        os.makedirs(MEDIA_DIR)
        logger.info(f"Created root media directory: {MEDIA_DIR}")
        
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS card_spaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at INTEGER NOT NULL, last_updated_at INTEGER NOT NULL)")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flashcards (
        id TEXT PRIMARY KEY, card_space_id TEXT NOT NULL, word_phrase TEXT NOT NULL, 
        image_filename TEXT, created_at INTEGER NOT NULL, last_updated_at INTEGER NOT NULL, 
        FOREIGN KEY (card_space_id) REFERENCES card_spaces (id) ON DELETE CASCADE, 
        UNIQUE (card_space_id, word_phrase)
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flashcard_fields (
        id TEXT PRIMARY KEY, flashcard_id TEXT NOT NULL, field_type TEXT NOT NULL, 
        content TEXT NOT NULL, llm_model_used TEXT, sort_order INTEGER DEFAULT 0, 
        FOREIGN KEY (flashcard_id) REFERENCES flashcards (id) ON DELETE CASCADE
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tutor_interactions (
        id TEXT PRIMARY KEY,
        flashcard_id TEXT NOT NULL,
        tutor_session_type TEXT NOT NULL, -- 'tricky_question', 'mcq', 'write_grade', 'synonym', 'antonym'
        llm_generated_content TEXT,     -- JSON: {question, options, correct_idx, explanation} for mcq; 
                                        --       {term, explanation, example} for syn/ant; 
                                        --       question_text for tricky_q
        user_response TEXT,             -- User's answer, choice index for mcq, or written text
        llm_feedback TEXT,              -- LLM's evaluation or feedback
        created_at INTEGER NOT NULL,
        FOREIGN KEY (flashcard_id) REFERENCES flashcards (id) ON DELETE CASCADE
    )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Flashcard DB '{DATABASE_NAME}' initialized/checked (with tutor_interactions table).")

def run_query(query, params=(), fetchone=False, fetchall=False, commit=False):
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    data = None
    try:
        cursor.execute(query, params)
        if fetchone: data = cursor.fetchone()
        if fetchall: data = cursor.fetchall()
        if commit: conn.commit()
    except sqlite3.Error as e:
        logger.error(f"DB Error: {e} for query '{query}' with params {params}", exc_info=True)
    finally:
        if conn: conn.close()
    return data

def load_app_settings():
    defaults = {"gemini_api_key": "", "ollama_endpoint": "http://localhost:11434", "ollama_model": "llama3.1:8b"}
    if os.path.exists(SETTINGS_FILE_PATH):
        try:
            with open(SETTINGS_FILE_PATH, 'r') as f: settings = json.load(f)
            for key_s in defaults: 
                if key_s in settings: defaults[key_s] = settings[key_s] # Update defaults with loaded settings
            logger.info(f"Loaded settings from {SETTINGS_FILE_PATH}")
            return settings # Return loaded settings merged with defaults
        except Exception as e: 
            logger.error(f"Error loading {SETTINGS_FILE_PATH}: {e}. Using defaults.")
    else: 
        logger.info(f"{SETTINGS_FILE_PATH} not found. Using default settings.")
    return defaults


def save_app_settings(settings_dict):
    try:
        with open(SETTINGS_FILE_PATH, 'w') as f: json.dump(settings_dict, f, indent=4)
        logger.info(f"Settings saved to {SETTINGS_FILE_PATH}")
    except Exception as e: logger.error(f"Error saving settings to {SETTINGS_FILE_PATH}: {e}")
