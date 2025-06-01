# main_flashcard_app.py
from llm_processors import OllamaLlamaProcessor, GeminiAPIProcessor
import streamlit as st
import html
import uuid
import os
import inspect # <<<< ADD THIS
from PIL import Image 
import io 
import time 
import json 
import sqlite3 
import httpx 
from google import genai
from google.genai import types as google_genai_types
import asyncio 
import logging 
import re

# --- Setup Logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# --- Constants ---
DATABASE_NAME = "flashcard_db.sqlite"
MEDIA_DIR = "flashcard_media" 
SETTINGS_FILE_PATH = "flash_app_settings.json" 

# --- Database Initialization and Utility ---
if not os.path.exists(MEDIA_DIR): 
    os.makedirs(MEDIA_DIR)
    logger.info(f"Created root media directory: {MEDIA_DIR}")

def init_flashcard_db():
    # ... (DB Init code as before) ...
    if not os.path.exists(MEDIA_DIR): os.makedirs(MEDIA_DIR)
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS card_spaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at INTEGER NOT NULL, last_updated_at INTEGER NOT NULL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS flashcards (id TEXT PRIMARY KEY, card_space_id TEXT NOT NULL, word_phrase TEXT NOT NULL, image_filename TEXT, created_at INTEGER NOT NULL, last_updated_at INTEGER NOT NULL, FOREIGN KEY (card_space_id) REFERENCES card_spaces (id) ON DELETE CASCADE, UNIQUE (card_space_id, word_phrase) )")
    cursor.execute("CREATE TABLE IF NOT EXISTS flashcard_fields (id TEXT PRIMARY KEY, flashcard_id TEXT NOT NULL, field_type TEXT NOT NULL, content TEXT NOT NULL, llm_model_used TEXT, sort_order INTEGER DEFAULT 0, FOREIGN KEY (flashcard_id) REFERENCES flashcards (id) ON DELETE CASCADE )")
    conn.commit()
    conn.close()
    logger.info(f"Flashcard DB '{DATABASE_NAME}' initialized/checked.")


def run_query(query, params=(), fetchone=False, fetchall=False, commit=False):
    # ... (run_query code as before) ...
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row; cursor = conn.cursor(); data = None
    try:
        cursor.execute(query, params)
        if fetchone: data = cursor.fetchone()
        if fetchall: data = cursor.fetchall()
        if commit: conn.commit()
    except sqlite3.Error as e: logger.error(f"DB Error: {e} for query '{query}' with params {params}")
    finally:
        if conn: conn.close()
    return data

def load_app_settings():
    # ... (load_app_settings code as before) ...
    defaults = {"gemini_api_key": "", "ollama_endpoint": "http://localhost:11434", "ollama_model": "llama3.1:8b"}
    if os.path.exists(SETTINGS_FILE_PATH):
        try:
            with open(SETTINGS_FILE_PATH, 'r') as f: settings = json.load(f)
            for key_s in defaults: 
                if key_s in settings: defaults[key_s] = settings[key_s]
            logger.info(f"Loaded settings from {SETTINGS_FILE_PATH}")
        except Exception as e: logger.error(f"Error loading {SETTINGS_FILE_PATH}: {e}. Using defaults.")
    else: logger.info(f"{SETTINGS_FILE_PATH} not found. Using default settings.")
    return defaults

def save_app_settings(settings_dict):
    # ... (save_app_settings code as before) ...
    try:
        with open(SETTINGS_FILE_PATH, 'w') as f: json.dump(settings_dict, f, indent=4)
        logger.info(f"Settings saved to {SETTINGS_FILE_PATH}")
    except Exception as e: logger.error(f"Error saving settings to {SETTINGS_FILE_PATH}: {e}")

# --- LLM Processor Classes ---

class GeminiAPIProcessor:
    def __init__(self, api_key: str):
        self.instance_id_log = str(id(self))[-6:]
        if not api_key:
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Initialization failed: API key is missing.")
            raise ValueError("Gemini API key is required for GeminiAPIProcessor.")
        
        try:
            self.client = genai.Client(api_key=api_key) 
            # Optional: list_models for validation
            # _ = list(self.client.list_models(page_size=1))
            logger.info(f"GeminiAPIProc Inst:{self.instance_id_log} GenAI client CREATED with API key.")
        except Exception as e:
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Failed to create GenAI client: {e}", exc_info=True)
            self.client = None 
            raise 
        
        self.default_model_name_str = "gemini-2.0-flash" # From your sample code
        self.generation_count = 0
        logger.info(f"GeminiAPIProc Inst:{self.instance_id_log} initialized. Default model string: {self.default_model_name_str}")

    async def _generate_content_with_gemini_using_client_models_stream(
        self, 
        model_name_str_to_use: str, 
        current_contents_for_api: list, # list of types.Content
        generation_config_obj: google_genai_types.GenerateContentConfig,
        # stream_to_client: bool = False # For future true streaming to client UI
    ):
        """
        Internal helper that uses the client.models.generate_content_stream pattern
        and collects the response.
        """
        loop = asyncio.get_event_loop()
        
        def sync_call_gemini_stream_and_collect():
            full_text_response = ""
            # This directly matches your sample's generation call structure
            response_stream = self.client.models.generate_content_stream(
                model=model_name_str_to_use, # e.g., "gemini-2.0-flash"
                contents=current_contents_for_api,
                config=generation_config_obj 
            )
            for chunk in response_stream:
                current_chunk_text = ""
                if hasattr(chunk, 'text') and chunk.text:
                    current_chunk_text = chunk.text
                elif chunk.parts: 
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            current_chunk_text += part.text
                if current_chunk_text:
                    full_text_response += current_chunk_text
            return full_text_response.strip()

        return await loop.run_in_executor(None, sync_call_gemini_stream_and_collect)

    async def generate_text(self, system_prompt: str, user_prompt: str, image_bytes: bytes = None, output_format_json: bool = False, rag_context: str = ""):
        self.generation_count += 1
        model_str_to_use = self.default_model_name_str
        log_prefix = f"GeminiAPIProc Inst:{self.instance_id_log} GenCall {self.generation_count} (ModelStr: {model_str_to_use}):"

        if not self.client:
            logger.error(f"{log_prefix} Gemini client not initialized.")
            return "[Error: Gemini client not initialized]"

        try:
            # Prepare user_parts (list of google_genai_types.Part)
            user_parts_list = []
            full_user_text_prompt = user_prompt
            if rag_context:
                full_user_text_prompt = f"{rag_context}\n\n---\n\nUser Query:\n{user_prompt}"
            
            parts_list = [google_genai_types.Part(text=full_user_text_prompt)]
            if image_bytes:
                try:
                    image_part_data = {"mime_type": "image/jpeg", "data": image_bytes}
                    parts_list.append(google_genai_types.Part(inline_data=image_part_data))
                except Exception as e_img:
                    logger.error(f"Error creating image part for Gemini chat: {e_img}")
            
            # Prepare contents (list of google_genai_types.Content)
            current_contents_for_api = [google_genai_types.Content(role="user", parts=parts_list)]

            # Prepare generation_config (google_genai_types.GenerateContentConfig)
            response_mime = "application/json" if output_format_json else "text/plain"
            gen_config_dict = {"response_mime_type": response_mime}
            if system_prompt:
                 # For client.models.generate_content_stream, system_instruction is part of config
                gen_config_dict["system_instruction"] = [google_genai_types.Part.from_text(text=system_prompt)]
            
            # Add tools if needed for this specific call, from your sample
            tools = [google_genai_types.Tool(google_search=google_genai_types.GoogleSearch())] if not output_format_json else None
            if tools:
                gen_config_dict["tools"] = tools

            generation_config_obj = google_genai_types.GenerateContentConfig(**gen_config_dict)

            logger.debug(f"{log_prefix} Calling _generate_content_with_gemini_using_client_models_stream. System: '{system_prompt[:50]}...', User: '{user_prompt[:50]}...', Image: {image_bytes is not None}")

            generated_text = await self._generate_content_with_gemini_using_client_models_stream(
                model_name_str_to_use=model_str_to_use,
                current_contents_for_api=current_contents_for_api,
                generation_config_obj=generation_config_obj
                # stream_to_client=False (this internal helper now always collects)
            )

            if output_format_json and generated_text and not generated_text.startswith("[Error:"):
                try: 
                    return json.loads(generated_text)
                except json.JSONDecodeError: 
                    logger.error(f"Gemini: Failed to parse JSON from: {generated_text}")
                    return {"error": "LLM JSON parse error", "raw_output": generated_text}
            
            return generated_text # Already stripped by the helper

        except Exception as e: # Catch errors from this public method's logic
            logger.error(f"{log_prefix} Error in generate_text setup: {e}", exc_info=True)
            return f"[Error: Gemini processing setup failed - {type(e).__name__}]"


    async def generate_eval_response(self, newest_summary: str, latest_user_query: str, latest_image_bytes: bytes = None):
        eval_system_instruction = """<EVAL>
You're a plant-disease expert, with great knowledge base. Your job is to provide an enhanced "clarification" answer that aims to resolve insufficient answering from a smaller model, based on a summarization of the chat session, the user's latest query, and the latest uploaded image. Output ONLY the enhanced answer.
"""
        formatted_eval_user_prompt = f"""Current Session Summary:
{newest_summary}

User's latest query that prompted the smaller model's response:
{latest_user_query}

[Image is also provided if attached to this message]

Based on the above, provide your enhanced answer.
"""
        eval_user_parts = [google_genai_types.Part(text=formatted_eval_user_prompt)]
        if latest_image_bytes:
            try:
                image_part_data_eval = {"mime_type": "image/jpeg", "data": latest_image_bytes}
                eval_user_parts.append(google_genai_types.Part(inline_data=image_part_data_eval))
            except Exception as e_img_eval: logger.error(f"Error creating image part for Gemini eval: {e_img_eval}")
        
        eval_tools = [google_genai_types.Tool(google_search=google_genai_types.GoogleSearch())]
        
        # For _generate_content_with_gemini_using_client_models_stream, we pass the model name string directly
        generated_text = await self._generate_content_with_gemini_using_client_models_stream(
            model_name_str_to_use="gemini-2.0-flash", # As per your eval sample
            current_contents_for_api=[google_genai_types.Content(role="user", parts=eval_user_parts)],
            generation_config_obj=google_genai_types.GenerateContentConfig(
                system_instruction=[google_genai_types.Part.from_text(text=eval_system_instruction)],
                tools=eval_tools,
                response_mime_type="text/plain"
            )
        )
        
        if generated_text and not generated_text.startswith("[Error:") and not generated_text.strip().startswith("<EVAL>"): 
            return f"<EVAL>\n{generated_text.strip()}"
        elif not generated_text or generated_text.startswith("[Error:"): 
            return "<EVAL>\n[Gemini Eval: No response generated or error occurred]"
        return generated_text


init_flashcard_db()
app_settings = load_app_settings()

if 'active_space_id' not in st.session_state: st.session_state.active_space_id = None
if 'gemini_api_key' not in st.session_state: st.session_state.gemini_api_key = app_settings.get("gemini_api_key", "")
if 'use_gemini' not in st.session_state: st.session_state.use_gemini = bool(st.session_state.gemini_api_key and st.session_state.gemini_api_key.strip())
if 'ollama_endpoint' not in st.session_state: st.session_state.ollama_endpoint = app_settings.get("ollama_endpoint", "http://localhost:11434")
if 'ollama_model' not in st.session_state: st.session_state.ollama_model = app_settings.get("ollama_model", "llama3.1:8b")
if 'current_view_mode' not in st.session_state: st.session_state.current_view_mode = 'list_cards'
if 'editing_flashcard_id' not in st.session_state: st.session_state.editing_flashcard_id = None
if 'studying_flashcard_id' not in st.session_state: st.session_state.studying_flashcard_id = None
if 'studying_deck_space_id' not in st.session_state: 
    st.session_state.studying_deck_space_id = None; st.session_state.study_deck_index = 0; st.session_state.study_card_flipped = False

field_types_config = {
    "definition": {
        "label": "Definition(s)", "json_output": False, "allow_multiple": True,
        "prompt_template": "Provide a clear and concise definition for the English word/phrase: '{WORD_PHRASE}'. If multiple common meanings exist, provide the primary ones. Output only the definition text.",
        "user_input_label": "Your Definition"
    },
    "word_family": {
        "label": "Word Family", "json_output": True, "allow_multiple": False, # Usually one comprehensive word family
        "prompt_template": "Generate the derivational word family for the English word/phrase: '{WORD_PHRASE}'. Include related nouns, verbs, adjectives, and adverbs that share the same morphological root. Output as a JSON object with keys like 'root_word' (string), 'nouns' (list of strings), 'verbs' (list of strings), 'adjectives' (list of strings), 'adverbs' (list of strings). Focus strictly on morphological derivations.",
        "user_input_label": "Your Word Family (JSON)",
        "template_for_user": json.dumps({
            "root_word": "",
            "nouns": [""],
            "verbs": [""],
            "adjectives": [""],
            "adverbs": [""]
        }, indent=4) # The default JSON string user will see
    },
    "example_sentence": {
        "label": "Example Sentence(s)", "json_output": False, "allow_multiple": True,
        "prompt_template": "Generate 1 distinct example sentence using the English word/phrase: '{WORD_PHRASE}'. Ensure it's suitable for an intermediate English learner and clearly demonstrates its usage. Output only the sentence.",
        "user_input_label": "Your Example Sentence"
    },
    "pairwise_dialogue": {
        "label": "Pairwise Dialogue (Q&A)", "json_output": True, "allow_multiple": True,
        "prompt_template": "Create a short, natural-sounding dialogue snippet (a question followed by an answer) for an English learner. The dialogue should use or clearly relate to the target English word/phrase: '{WORD_PHRASE}'. Output as a JSON object with two keys: \"question\" (string) and \"answer\" (string).",
        "user_input_label": "Your Q&A (JSON)",
        "template_for_user": json.dumps({
            "question": "",
            "answer": ""
        }, indent=4) # The default JSON string user will see
    }
}
def initialize_llm_processors():
    logger.info("--- Running initialize_llm_processors ---")
    try:
        ollama_needs_init = 'ollama_llm_instance' not in st.session_state or st.session_state.ollama_llm_instance is None or \
                           (hasattr(st.session_state.ollama_llm_instance, 'base_url') and st.session_state.ollama_llm_instance.base_url != st.session_state.ollama_endpoint) or \
                           (hasattr(st.session_state.ollama_llm_instance, 'model_name') and st.session_state.ollama_llm_instance.model_name != st.session_state.ollama_model)
        if ollama_needs_init:
            logger.info(f"Initializing/Re-initializing Ollama: {st.session_state.ollama_endpoint}, {st.session_state.ollama_model}")
            logger.info(f"OllamaLlamaProcessor class source: {inspect.getfile(OllamaLlamaProcessor)}")
            st.session_state.ollama_llm_instance = OllamaLlamaProcessor(base_url=st.session_state.ollama_endpoint, model_name=st.session_state.ollama_model)
            logger.info(f"Ollama processor initialized. Type: {type(st.session_state.ollama_llm_instance)}")
            if hasattr(st.session_state.ollama_llm_instance, 'generate_text'):
                sig = inspect.signature(st.session_state.ollama_llm_instance.generate_text)
                logger.info(f"Signature of ollama_llm_instance.generate_text: {sig}")

    except Exception as e:
        logger.error(f"Failed to init Ollama: {e}", exc_info=True)
        st.session_state.ollama_llm_instance = None

    gemini_needs_init = True
    if 'gemini_llm_instance' in st.session_state and st.session_state.gemini_llm_instance is not None:
        # Simplified check for re-initialization for Gemini based on API key change
        # Assuming Gemini processor internal state doesn't change other than by API key for now
        # The previous check was a bit complex and might not be fully robust for genai.Client
        current_instance_api_key_check = getattr(st.session_state.gemini_llm_instance.client, '_client_options.api_key', None) if hasattr(st.session_state.gemini_llm_instance, 'client') and st.session_state.gemini_llm_instance.client else None
        if current_instance_api_key_check == st.session_state.gemini_api_key and st.session_state.gemini_api_key and st.session_state.gemini_api_key.strip():
             gemini_needs_init = False
             logger.info("Gemini processor already initialized with the current API key. Skipping re-init.")


    if st.session_state.gemini_api_key and st.session_state.gemini_api_key.strip():
        if gemini_needs_init:
            try:
                logger.info("Attempting to initialize/re-initialize Gemini processor.")
                logger.info(f"GeminiAPIProcessor class source: {inspect.getfile(GeminiAPIProcessor)}")
                st.session_state.gemini_llm_instance = GeminiAPIProcessor(api_key=st.session_state.gemini_api_key)
                logger.info(f"Gemini processor initialized. Type: {type(st.session_state.gemini_llm_instance)}")
                if hasattr(st.session_state.gemini_llm_instance, 'generate_text'):
                    sig = inspect.signature(st.session_state.gemini_llm_instance.generate_text)
                    logger.info(f"Signature of st.session_state.gemini_llm_instance.generate_text: {sig}")
                else:
                    logger.error("CRITICAL: Initialized gemini_llm_instance is MISSING generate_text method!")
            except Exception as e:
                logger.error(f"Failed to init Gemini: {e}", exc_info=True)
                st.session_state.gemini_llm_instance = None
    elif st.session_state.get('gemini_llm_instance') is not None:
        logger.info("Gemini API key removed/empty, clearing Gemini processor instance.")
        st.session_state.gemini_llm_instance = None
    logger.info("--- Finished initialize_llm_processors ---")

if 'llm_processors_initialized_flag' not in st.session_state:
    initialize_llm_processors()
    st.session_state.llm_processors_initialized_flag = True

def get_active_llm_processor():
    logger.debug("--- Running get_active_llm_processor ---")
    active_llm = None
    if st.session_state.get('use_gemini', False):
        active_llm = st.session_state.get('gemini_llm_instance')
        logger.debug(f"Gemini selected. Instance from session_state: {type(active_llm)}")
        if not active_llm and st.session_state.get('gemini_api_key'):
            logger.warning("Gemini selected, but instance is None. Attempting re-init via initialize_llm_processors().")
            initialize_llm_processors() # Try to re-initialize
            active_llm = st.session_state.get('gemini_llm_instance')
            logger.debug(f"Gemini instance after re-init attempt: {type(active_llm)}")
        if not active_llm:
            st.sidebar.warning("Gemini selected, but processor failed. Check API Key and Save.")
        elif hasattr(active_llm, 'generate_text'): # Log signature if instance is valid
            sig = inspect.signature(active_llm.generate_text)
            logger.debug(f"Returning active_llm (Gemini). generate_text signature: {sig}")
        else:
            logger.error(f"Active Gemini LLM instance ({type(active_llm)}) is MISSING generate_text method!")

    else: # Ollama
        active_llm = st.session_state.get('ollama_llm_instance')
        logger.debug(f"Ollama selected. Instance from session_state: {type(active_llm)}")
        if not active_llm:
            logger.warning("Ollama selected, but instance is None. Attempting re-init via initialize_llm_processors().")
            initialize_llm_processors() # Try to re-initialize
            active_llm = st.session_state.get('ollama_llm_instance')
            logger.debug(f"Ollama instance after re-init attempt: {type(active_llm)}")
        if not active_llm:
            st.sidebar.warning("Ollama selected, but processor failed. Check settings.")
        elif hasattr(active_llm, 'generate_text'): # Log signature if instance is valid
            sig = inspect.signature(active_llm.generate_text)
            logger.debug(f"Returning active_llm (Ollama). generate_text signature: {sig}")
        else:
             logger.error(f"Active Ollama LLM instance ({type(active_llm)}) is MISSING generate_text method!")
    
    logger.debug(f"--- get_active_llm_processor returning: {type(active_llm)} ---")
    return active_llm

async def generate_with_active_llm(system_prompt, user_prompt, image_bytes=None, output_json=False): # output_json parameter name is fine here
    active_llm = get_active_llm_processor()
    
    logger.debug(f"[generate_with_active_llm] After get_active_llm_processor: type(active_llm) is {type(active_llm)}")
    if active_llm:
        logger.debug(f"[generate_with_active_llm] Attributes of active_llm: {dir(active_llm)}")

    if not active_llm:
        st.error("LLM Processor is not available. Please check settings.")
        logger.error("generate_with_active_llm: get_active_llm_processor returned None.")
        return None 
    
    is_gemini_selected_in_session = st.session_state.get('use_gemini', False)
    logger.debug(f"[generate_with_active_llm] Dispatching. use_gemini toggle: {is_gemini_selected_in_session}")

    try:
        if is_gemini_selected_in_session:
            if hasattr(active_llm, 'generate_text'):
                logger.info(f"Calling GeminiAPIProcessor.generate_text with model: {getattr(active_llm, 'default_model_name_str', 'N/A')}")
                # Ensure rag_context is passed if GeminiAPIProcessor expects it (the one from main_flashcard_app did)
                return await active_llm.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_bytes=image_bytes,
                    output_format_json=output_json,  # CHANGED: use output_format_json
                    rag_context=""  # Assuming rag_context is handled or empty for flashcards
                )
            else:
                st.error("Gemini selected, but 'generate_text' method is missing.")
                logger.error(f"AttributeMissing: Gemini selected. active_llm (type: {type(active_llm)}) is missing 'generate_text'. dir(): {dir(active_llm)}")
                return None
        else: # Ollama should be used
            if hasattr(active_llm, 'generate_text'):
                logger.info(f"Calling OllamaLlamaProcessor.generate_text with model: {getattr(active_llm, 'model_name', 'N/A')}")
                prompt_to_send = user_prompt
                if image_bytes: 
                    prompt_to_send = f"{user_prompt}\n\n[Context: User has also uploaded an image.]" # Ollama doesn't directly take image_bytes
                
                return await active_llm.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=prompt_to_send,
                    output_format_json=output_json  # CHANGED: use output_format_json
                )
            else:
                st.error("Ollama selected, but 'generate_text' method is missing.")
                logger.error(f"AttributeMissing: Ollama selected. active_llm (type: {type(active_llm)}) is missing 'generate_text'. dir(): {dir(active_llm)}")
                return None
    except Exception as e:
        # Log the specific type of active_llm for better debugging if needed
        llm_type_info = type(active_llm).__name__ if active_llm else "None"
        st.error(f"An unexpected error occurred during the LLM call ({llm_type_info}): {type(e).__name__} - {e}")
        logger.error(f"Exception in LLM call ({llm_type_info}) within generate_with_active_llm: {e}", exc_info=True)
        return None

# --- Rendering Functions (Defined before main UI logic that calls them) ---
def render_list_flashcards_view(space_id: str):
    # ... (Full implementation from previous responses - with st.rerun() and unique keys) ...
    with st.expander("➕ Add New Flashcard Stub", expanded=True):
        with st.form("new_flashcard_stub_form", clear_on_submit=True):
            word_phrase_input = st.text_input("Word or Phrase for New Card*", key="fc_stub_word_list_view")
            submitted_stub = st.form_submit_button("Create Flashcard Stub & Edit")
            if submitted_stub and word_phrase_input.strip():
                existing_card = run_query("SELECT id FROM flashcards WHERE card_space_id = ? AND lower(word_phrase) = ?", (space_id, word_phrase_input.strip().lower()), fetchone=True)
                if existing_card: st.warning(f"Flashcard '{word_phrase_input.strip()}' already exists.")
                else: flashcard_id = str(uuid.uuid4()); now = int(time.time()); run_query("INSERT INTO flashcards (id, card_space_id, word_phrase, created_at, last_updated_at) VALUES (?, ?, ?, ?, ?)", (flashcard_id, space_id, word_phrase_input.strip(), now, now), commit=True); st.success(f"Stub '{word_phrase_input.strip()}' created."); st.session_state.editing_flashcard_id = flashcard_id; st.session_state.current_view_mode = 'edit_card'; st.rerun()
            elif submitted_stub: st.warning("Word/Phrase cannot be empty.")
    st.markdown("---")
    if st.button("📚 Study All Cards in this Space", key=f"study_all_btn_list_view_{space_id}"):
        flashcards_check = run_query("SELECT id FROM flashcards WHERE card_space_id = ?", (space_id,), fetchall=True)
        if flashcards_check: st.session_state.studying_deck_space_id = space_id; st.session_state.study_deck_index = 0; st.session_state.study_card_flipped = False; st.session_state.current_view_mode = 'study_deck'; st.rerun()
        else: st.warning("No cards to study.")
    flashcards_in_space = run_query("SELECT id, word_phrase FROM flashcards WHERE card_space_id = ? ORDER BY created_at DESC", (space_id,), fetchall=True)
    if not flashcards_in_space: st.info("No flashcards in this space yet."); return
    st.subheader("Your Flashcards:")
    for card_data_row in flashcards_in_space:
        card_id = card_data_row['id']; word_phrase = card_data_row['word_phrase']
        col1, col2, col3, col4 = st.columns([0.5, 0.17, 0.17, 0.16])
        with col1: st.markdown(f"**{word_phrase}**")
        with col2:
            if st.button("✏️ Edit", key=f"edit_btn_list_item_render_{card_id}"): st.session_state.editing_flashcard_id = card_id; st.session_state.current_view_mode = 'edit_card'; st.rerun()
        with col3:
            if st.button("📖 Study", key=f"study_single_btn_list_item_render_{card_id}"):
                st.session_state.studying_flashcard_id = card_id
                st.session_state.current_view_mode = 'study_card' # CRITICAL
                st.session_state.study_card_flipped = False # Reset flip state
                # Ensure deck study states are cleared to avoid conflict
                st.session_state.studying_deck_space_id = None 
                st.rerun()
        with col4: 
            if f"confirm_delete_card_state_list_render_{card_id}" not in st.session_state: st.session_state[f"confirm_delete_card_state_list_render_{card_id}"] = False
            if st.session_state[f"confirm_delete_card_state_list_render_{card_id}"]:
                if st.button("✅ Confirm Delete", key=f"confirm_del_btn_list_item_render_{card_id}", type="primary"): run_query("DELETE FROM flashcards WHERE id = ?", (card_id,), commit=True); st.success(f"Card '{word_phrase}' deleted."); st.session_state[f"confirm_delete_card_state_list_render_{card_id}"] = False; st.rerun()
                if st.button("❌ Cancel", key=f"cancel_del_btn_list_item_render_{card_id}"): st.session_state[f"confirm_delete_card_state_list_render_{card_id}"] = False; st.rerun()
            else:
                if st.button("🗑️ Del", key=f"del_card_btn_list_item_render_{card_id}", help="Delete card"):  st.session_state[f"confirm_delete_card_state_list_render_{card_id}"] = True; st.rerun()
        st.divider()

def render_edit_flashcard_view(card_id: str, space_id: str):
    # ... (top part of the function remains the same: card_data, image handling ...)
    card_data = run_query("SELECT id, word_phrase, image_filename FROM flashcards WHERE id = ? AND card_space_id = ?", (card_id, space_id), fetchone=True)
    if not card_data: st.error("Flashcard not found."); st.session_state.current_view_mode = 'list_cards'; st.session_state.editing_flashcard_id = None; st.rerun(); return
    st.subheader(f"✏️ Editing Card: {html.escape(card_data['word_phrase'])}")
    current_image_filename = card_data['image_filename']; image_bytes_for_llm_card_level = None 
    if current_image_filename:
        img_full_path = os.path.join(MEDIA_DIR, space_id, current_image_filename); 
        if os.path.exists(img_full_path): st.image(img_full_path, caption=current_image_filename, width=150); 
        try: 
            with open(img_full_path, "rb") as f_img: image_bytes_for_llm_card_level = f_img.read()
        except Exception as e_read: logger.error(f"Error reading image {img_full_path} for LLM: {e_read}")
        if st.button("Remove Image", key=f"remove_img_btn_edit_view_render_{card_id}"): 
            try: 
                if os.path.exists(img_full_path): os.remove(img_full_path)
                run_query("UPDATE flashcards SET image_filename = NULL, last_updated_at = ? WHERE id = ?", (int(time.time()), card_id), commit=True); st.rerun()
            except OSError as e_os: st.error(f"Error removing image file: {e_os}")
    uploaded_image_file_edit = st.file_uploader("Change/Upload Image", type=["jpg", "jpeg", "png"], key=f"edit_img_uploader_widget_view_render_{card_id}")
    if uploaded_image_file_edit:
        space_media_dir = os.path.join(MEDIA_DIR, space_id); os.makedirs(space_media_dir, exist_ok=True)
        if current_image_filename: 
            old_img_path = os.path.join(space_media_dir, current_image_filename); 
            if os.path.exists(old_img_path): 
                try: os.remove(old_img_path); 
                except OSError as e: st.warning(f"Could not remove old: {e}")
        new_image_filename = f"{card_id}_{uploaded_image_file_edit.name}"; image_save_path = os.path.join(space_media_dir, new_image_filename)
        with open(image_save_path, "wb") as f: f.write(uploaded_image_file_edit.getbuffer())
        run_query("UPDATE flashcards SET image_filename = ?, last_updated_at = ? WHERE id = ?", (new_image_filename, int(time.time()), card_id), commit=True); st.success("Image updated!"); st.rerun()
    st.markdown("---")
    card_fields_data = run_query("SELECT id, field_type, content, llm_model_used FROM flashcard_fields WHERE flashcard_id = ? ORDER BY sort_order ASC", (card_id,), fetchall=True)
    
    for field_key_loop, config_loop in field_types_config.items():
        with st.expander(f"{config_loop['label']}", expanded=True):
            existing_contents_loop = [f_row for f_row in card_fields_data if f_row['field_type'] == field_key_loop]
            for i, content_entry_loop in enumerate(existing_contents_loop):
                col1, col2 = st.columns([0.85, 0.15]); 
                with col1:
                    display_content_loop = content_entry_loop['content']
                    llm_used_info = content_entry_loop['llm_model_used'] or 'Manual'
                    if config_loop['json_output']:
                        try: 
                            # Pretty print JSON for display
                            parsed_json = json.loads(display_content_loop)
                            st.json(parsed_json)
                            st.caption(f"*(LLM: {llm_used_info})*")
                        except json.JSONDecodeError: 
                            st.code(display_content_loop, language='text') # Show as plain text if not valid JSON
                            st.caption(f"*(LLM: {llm_used_info} - Invalid JSON format)*")
                    else: 
                        st.markdown(f"- {html.escape(display_content_loop)} *(LLM: {llm_used_info})*")
                with col2:
                    if st.button("🗑️", key=f"del_edit_field_btn_view_item_render_{field_key_loop}_{content_entry_loop['id']}", help="Delete entry"): 
                        run_query("DELETE FROM flashcard_fields WHERE id = ?", (content_entry_loop['id'],), commit=True); st.rerun()
            
            # Form for adding new user entry
            with st.form(key=f"form_user_edit_add_view_item_render_{field_key_loop}_{card_id}", clear_on_submit=True):
                # Use the template if available for this field type
                default_user_input = config_loop.get('template_for_user', "")
                
                # Determine height based on content type
                text_area_height = 150 if config_loop.get('json_output', False) else 75

                user_field_input_loop = st.text_area(
                    label=f"Your {config_loop['label'].split('(')[0].strip()}",
                    value=default_user_input, # Pre-fill with template
                    height=text_area_height, 
                    key=f"user_edit_input_ta_view_item_render_{field_key_loop}_{card_id}"
                )

                if st.form_submit_button(f"Add My Entry"):
                    if user_field_input_loop.strip():
                        # Validate JSON if it's supposed to be JSON
                        if config_loop.get('json_output', False):
                            try:
                                json.loads(user_field_input_loop.strip()) # Try to parse
                            except json.JSONDecodeError:
                                st.error("Invalid JSON format. Please correct it or ensure it matches the template structure.")
                                st.stop() # Prevent saving invalid JSON

                        field_id_loop = str(uuid.uuid4())
                        run_query("INSERT INTO flashcard_fields (id, flashcard_id, field_type, content, llm_model_used, sort_order) VALUES (?, ?, ?, ?, ?, ?)", 
                                  (field_id_loop, card_id, field_key_loop, user_field_input_loop.strip(), "user_manual", len(existing_contents_loop)), 
                                  commit=True)
                        st.success(f"Your entry added!")
                        st.rerun()
                    else:
                        st.warning("Input cannot be empty.")

            # LLM Generation Button
            # Condition to show "Generate with LLM" button:
            # - If multiple entries are allowed OR
            # - If no LLM-generated entry of this type already exists (for fields where only one LLM entry makes sense, like word_family)
            show_llm_button = config_loop['allow_multiple'] or \
                              not any(ec['llm_model_used'] != "user_manual" and ec['llm_model_used'] is not None and ec['field_type'] == field_key_loop for ec in existing_contents_loop)

            if show_llm_button:
                if st.button(f"Generate with LLM", key=f"gen_llm_edit_btn_view_item_render_{field_key_loop}_{card_id}"):
                    active_llm_proc_loop = get_active_llm_processor()
                    if not active_llm_proc_loop: 
                        st.warning("LLM processor not available. Check settings.")
                    else:
                        with st.spinner(f"Generating {config_loop['label']}..."):
                            user_prompt_for_llm_loop = config_loop['prompt_template'].replace("{WORD_PHRASE}", card_data['word_phrase'])
                            
                            # Decide if image should be sent for this field type
                            current_image_for_llm_field_loop = image_bytes_for_llm_card_level if field_key_loop in ["example_sentence", "definition", "pairwise_dialogue"] else None
                                                        
                            generated_content_loop = None; loop = None
                            try:
                                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                                generated_content_loop = loop.run_until_complete(generate_with_active_llm(
                                    system_prompt="You are an expert English language learning assistant.", 
                                    user_prompt=user_prompt_for_llm_loop, 
                                    image_bytes=current_image_for_llm_field_loop, 
                                    output_json=config_loop['json_output']
                                ))
                            except Exception as e_async_llm: 
                                logger.error(f"Async LLM call error for {field_key_loop}: {e_async_llm}", exc_info=True)
                                st.error(f"LLM call failed: {e_async_llm}")
                            finally:
                                if loop is not None: 
                                    # Ensure loop is properly closed
                                    if loop.is_running():
                                        # Attempt to cancel all tasks and stop the loop
                                        for task in asyncio.all_tasks(loop=loop):
                                            task.cancel()
                                        try:
                                            # Wait for tasks to cancel
                                            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop=loop), return_exceptions=True))
                                        except RuntimeError as e_runtime: # If loop is already closing/closed
                                            logger.warning(f"RuntimeError during loop cleanup for {field_key_loop}: {e_runtime}")
                                        except Exception as e_cleanup:
                                             logger.error(f"Exception during loop cleanup for {field_key_loop}: {e_cleanup}")
                                    
                                    if not loop.is_closed():
                                        loop.close()
                                        logger.debug(f"Event loop for {field_key_loop} explicitly closed.")
                                    else:
                                         logger.debug(f"Event loop for {field_key_loop} was already closed.")


                            if generated_content_loop:
                                if config_loop['json_output'] and isinstance(generated_content_loop, dict) and generated_content_loop.get("error"): 
                                    st.error(f"LLM JSON Error: {generated_content_loop.get('raw_output', 'Could not generate valid JSON.')}")
                                elif config_loop['json_output'] and not isinstance(generated_content_loop, dict):
                                     st.error(f"LLM did not return valid JSON for {config_loop['label']}. Received: {str(generated_content_loop)[:200]}")
                                else: 
                                    content_to_save_loop = json.dumps(generated_content_loop, indent=4) if config_loop['json_output'] and isinstance(generated_content_loop, dict) else str(generated_content_loop)
                                    field_id_llm_loop = str(uuid.uuid4())
                                    model_used_loop = "Gemini" if st.session_state.use_gemini and st.session_state.get('gemini_llm_instance') else st.session_state.ollama_model
                                    run_query("INSERT INTO flashcard_fields (id, flashcard_id, field_type, content, llm_model_used, sort_order) VALUES (?, ?, ?, ?, ?, ?)", 
                                              (field_id_llm_loop, card_id, field_key_loop, content_to_save_loop, model_used_loop, len(existing_contents_loop)), 
                                              commit=True)
                                    st.success(f"LLM generated entry added for {config_loop['label']}!"); 
                                    st.rerun()
                            elif not ('e_async_llm' in locals() and e_async_llm): # If no error was explicitly caught but no content
                                st.error(f"LLM failed to generate for {config_loop['label']} (no content returned).")
    
    if st.button("Done Editing (Back to List)", key=f"done_edit_back_to_list_btn_view_render_{card_id}"): 
        st.session_state.current_view_mode = 'list_cards'; st.session_state.editing_flashcard_id = None; st.rerun()

def render_study_single_card_view(card_id: str, space_id: str): # << THIS IS THE FUNCTION WE ARE FOCUSING ON
    logger.info(f"Rendering SINGLE CARD view for card_id: {card_id}") # Add this log
    card_data = run_query("SELECT id, word_phrase, image_filename FROM flashcards WHERE id = ? AND card_space_id = ?", (card_id, space_id), fetchone=True)
    if not card_data:
        st.error("Flashcard for study not found.")
        st.session_state.current_view_mode = 'list_cards'
        st.session_state.studying_flashcard_id = None
        st.rerun(); return

    # 1. Header as per sketch
    st.subheader(f"📖 Studying Card: {html.escape(card_data['word_phrase'])}")
    st.markdown("---")

    if 'study_card_flipped' not in st.session_state or st.session_state.studying_flashcard_id != card_id:
        st.session_state.study_card_flipped = False
        st.session_state.studying_flashcard_id = card_id

    # 2. Main Area: Two Columns (Flippable Card | Image)
    col_card_area, col_image_area = st.columns([0.65, 0.35]) 

    with col_card_area:
        card_container_height = "450px" 
        
        if not st.session_state.study_card_flipped:
            front_html_style = f"""
                border: 2px solid #718096; background-color: #FFFFFF; color: #2D3748; padding: 20px;
                text-align: center; height: {card_container_height}; display: flex; flex-direction: column; 
                justify-content: center; align-items: center; border-radius: 10px;
                font-size: 2.8em; font-weight: bold; box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            """
            st.markdown(f"<div style='{front_html_style}'>{html.escape(card_data['word_phrase'])}</div>", unsafe_allow_html=True)
        else:
            back_html_style = f"""
                border: 2px solid #38B2AC; background-color: #F7FAFC; color: #2D3748; padding: 20px; 
                height: {card_container_height}; border-radius: 10px; overflow-y: auto; 
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            """
            back_html_content = f"<h4 style='margin-top:0; color: #2C5282; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px;'>{html.escape(card_data['word_phrase'])} - Details:</h4>"
            
            card_fields = run_query("SELECT field_type, content FROM flashcard_fields WHERE flashcard_id = ? ORDER BY field_type, sort_order ASC", (card_id,), fetchall=True)
            field_content_map = {}
            for field_row in card_fields:
                if field_row['field_type'] not in field_content_map: field_content_map[field_row['field_type']] = []
                field_content_map[field_row['field_type']].append(field_row['content'])

            for field_key_local, config_local in field_types_config.items():
                contents = field_content_map.get(field_key_local, [])
                if contents:
                    back_html_content += f"<div style='margin-top: 15px;'><strong style='color: #4A5568; font-size: 1.1em;'>{html.escape(config_local['label'])}:</strong></div><ul style='margin-top: 5px; margin-bottom:15px; padding-left: 25px; list-style-type: none;'>"
                    for entry_content_str in contents:
                        content_display = entry_content_str
                        item_style = "background-color: #EDF2F7; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"
                        if field_key_local == "pairwise_dialogue":
                            try:
                                dialogue_data = json.loads(content_display)
                                q = html.escape(dialogue_data.get('question', 'N/A'))
                                a = html.escape(dialogue_data.get('answer', 'N/A'))
                                content_display = f"<div style='{item_style}'><strong style='color:#3182CE;'>Q:</strong> {q}<br><strong style='color:#3182CE;'>A:</strong> {a}</div>"
                            except Exception as e: content_display = f"<div style='{item_style} background-color: #FFF5F5; color: #C53030;'><i>Invalid dialogue:</i> <pre>{html.escape(content_display)}</pre></div>"; logger.error(f"Error displaying pairwise_dialogue: {e}")
                        elif field_key_local == "word_family":
                            try:
                                parsed_json = json.loads(content_display)
                                wf_html = ""
                                for key, value_list in parsed_json.items():
                                    disp_key = html.escape(key.replace('_', ' ')).capitalize()
                                    if isinstance(value_list, list) and value_list and any(str(v).strip() for v in value_list): # Check if any item in list is non-empty after stripping
                                        wf_html += f"<div style='margin-bottom:3px;'><strong style='color:#007A5E;'>{disp_key}:</strong> {', '.join(html.escape(str(v)) for v in value_list if str(v).strip())}</div>"
                                    elif isinstance(value_list, str) and value_list.strip():
                                         wf_html += f"<div style='margin-bottom:3px;'><strong style='color:#007A5E;'>{disp_key}:</strong> {html.escape(value_list)}</div>"
                                no_terms_span = "<span style='font-style:italic; color: #718096;'>No terms provided.</span>"
                                inner_html_for_wf = wf_html if wf_html else no_terms_span
                                content_display = f"<div style='{item_style} background-color: #E6FFFA;'>{inner_html_for_wf}</div>"
                            except Exception as e_wf: content_display = f"<div style='{item_style} background-color: #FFF5F5; color: #C53030;'><i>Invalid Word Family JSON:</i> <pre>{html.escape(content_display)}</pre></div>"; logger.error(f"Error displaying word_family: {e_wf}")
                        elif config_local['json_output']: 
                             try:
                                parsed_json = json.loads(content_display)
                                content_display = f"<pre style='{item_style} white-space: pre-wrap; word-wrap: break-word; background-color: #F0F0F0;'>{html.escape(json.dumps(parsed_json, indent=2))}</pre>"
                             except Exception as e: content_display = f"<pre style='{item_style} white-space: pre-wrap; word-wrap: break-word; background-color: #FFF5F5; color: #C53030;'><i>Invalid JSON:</i> {html.escape(content_display)}</pre>"; logger.error(f"Error displaying other JSON: {e}")
                        else: 
                            content_display = f"<div style='{item_style}'>{html.escape(content_display).replace(chr(10), '<br>')}</div>"
                        back_html_content += f"<li>{content_display}</li>"
                    back_html_content += "</ul>"
            st.markdown(f"<div style='{back_html_style}'>{back_html_content}</div>", unsafe_allow_html=True)

    with col_image_area:
        if card_data['image_filename']:
            img_full_path = os.path.join(MEDIA_DIR, space_id, card_data['image_filename'])
            if os.path.exists(img_full_path):
                st.image(img_full_path, use_container_width=True, caption="Attached Image")
            else:
                logger.warning(f"Image file not found: {img_full_path}")
                st.markdown(f"<div style='height: {card_container_height}; display:flex; align-items:center; justify-content:center; border: 2px dashed #CBD5E0; border-radius:10px; color: #A0AEC0; text-align:center;'>Image not found<br><small>{html.escape(card_data['image_filename'])}</small></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='height: {card_container_height}; display:flex; align-items:center; justify-content:center; border: 2px dashed #CBD5E0; border-radius:10px; color: #A0AEC0;'>No Image Attached</div>", unsafe_allow_html=True)

    st.markdown("---") 

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Flip Card 🃏", key=f"flip_single_study_btn_view_render_{card_id}", use_container_width=True):
            st.session_state.study_card_flipped = not st.session_state.study_card_flipped
            st.rerun()
    with col_b2:
        if st.button("Back to Card List", key=f"study_single_back_to_list_btn_view_render_{card_id}", use_container_width=True):
            st.session_state.current_view_mode = 'list_cards'
            st.session_state.studying_flashcard_id = None
            st.session_state.study_card_flipped = False
            st.rerun()

    st.markdown("---") 

    # 4. Live Tutor Section
    st.subheader("🎓 Live Tutor") # THIS IS THE LIVE TUTOR SECTION
    
    tutor_state_key_base = f"tutor_state_{card_id}"
    if f"{tutor_state_key_base}_active_tab" not in st.session_state:
        st.session_state[f"{tutor_state_key_base}_active_tab"] = "Tricky Questions"

    tab1, tab2, tab3 = st.tabs(["🎯 Tricky Questions", "📝 MCQs", "✍️ Write & Grade"])

    with tab1: # Tricky Questions Tab
        st.markdown("#### Practice with Tricky Questions")
        if f"{tutor_state_key_base}_tricky_q" not in st.session_state:
            st.session_state[f"{tutor_state_key_base}_tricky_q"] = None
            st.session_state[f"{tutor_state_key_base}_tricky_a_feedback"] = None

        if st.button("Get a Tricky Question", key=f"tutor_get_tricky_q_{card_id}"):
            # TODO: LLM call here
            st.session_state[f"{tutor_state_key_base}_tricky_q"] = f"LLM will generate a tricky question about '{html.escape(card_data['word_phrase'])}'."
            st.session_state[f"{tutor_state_key_base}_tricky_a_feedback"] = None 
            st.rerun()

        if st.session_state[f"{tutor_state_key_base}_tricky_q"]:
            st.info(st.session_state[f"{tutor_state_key_base}_tricky_q"])
            user_answer_tricky = st.text_area("Your Answer:", key=f"tutor_user_answer_tricky_{card_id}", height=100)
            if st.button("Check My Answer", key=f"tutor_check_tricky_q_{card_id}"):
                if user_answer_tricky.strip():
                    # TODO: LLM call here
                    st.session_state[f"{tutor_state_key_base}_tricky_a_feedback"] = "LLM will provide feedback on your answer."
                    st.rerun()
                else: st.warning("Please provide an answer.")
            if st.session_state[f"{tutor_state_key_base}_tricky_a_feedback"]:
                st.success(st.session_state[f"{tutor_state_key_base}_tricky_a_feedback"])

    with tab2: # MCQs Tab
        st.markdown("#### Practice with MCQs")
        # ... (MCQ placeholder logic as provided in previous complete function) ...
        if f"{tutor_state_key_base}_mcq_data" not in st.session_state:
            st.session_state[f"{tutor_state_key_base}_mcq_data"] = None 
            st.session_state[f"{tutor_state_key_base}_mcq_user_choice"] = None
            st.session_state[f"{tutor_state_key_base}_mcq_feedback"] = None

        if st.button("Generate MCQ", key=f"tutor_get_mcq_{card_id}"):
            # TODO: LLM call here
            st.session_state[f"{tutor_state_key_base}_mcq_data"] = {
                "question": f"What is a synonym for '{html.escape(card_data['word_phrase'])}' in a specific context? (LLM generated)",
                "options": ["Option A (LLM)", "Option B (LLM)", "Option C (LLM)", "Option D (LLM)"],
                "correct_answer_idx": 0, 
                "explanation": "LLM will provide an explanation here."
            }
            st.session_state[f"{tutor_state_key_base}_mcq_user_choice"] = None
            st.session_state[f"{tutor_state_key_base}_mcq_feedback"] = None
            st.rerun()

        if st.session_state[f"{tutor_state_key_base}_mcq_data"]:
            mcq = st.session_state[f"{tutor_state_key_base}_mcq_data"]
            st.write(mcq["question"])
            if st.session_state[f"{tutor_state_key_base}_mcq_user_choice"] is not None and \
               st.session_state[f"{tutor_state_key_base}_mcq_data"] != st.session_state.get(f"{tutor_state_key_base}_mcq_data_prev_for_choice_reset"):
                st.session_state[f"{tutor_state_key_base}_mcq_user_choice"] = None

            user_choice_idx = st.radio(
                "Choose your answer:", 
                options=list(range(len(mcq["options"]))), 
                format_func=lambda idx: mcq["options"][idx],
                key=f"tutor_mcq_radio_{card_id}_{mcq['question'][:20]}", 
                index=st.session_state[f"{tutor_state_key_base}_mcq_user_choice"] # Persist selection
            )
            st.session_state[f"{tutor_state_key_base}_mcq_data_prev_for_choice_reset"] = mcq 

            if user_choice_idx is not None: 
                if st.button("Check MCQ Answer", key=f"tutor_check_mcq_{card_id}"):
                    st.session_state[f"{tutor_state_key_base}_mcq_user_choice"] = user_choice_idx # Record choice before feedback
                    # TODO: LLM could provide more dynamic feedback here
                    if user_choice_idx == mcq["correct_answer_idx"]:
                        st.session_state[f"{tutor_state_key_base}_mcq_feedback"] = f"Correct! 🎉\nExplanation: {mcq['explanation']}"
                    else:
                        st.session_state[f"{tutor_state_key_base}_mcq_feedback"] = f"Not quite. The correct answer was: {mcq['options'][mcq['correct_answer_idx']]}.\nExplanation: {mcq['explanation']}"
                    st.rerun()

            if st.session_state[f"{tutor_state_key_base}_mcq_feedback"]:
                is_correct = st.session_state.get(f"{tutor_state_key_base}_mcq_user_choice") == st.session_state.get(f"{tutor_state_key_base}_mcq_data",{}).get("correct_answer_idx")
                if is_correct:
                    st.success(st.session_state[f"{tutor_state_key_base}_mcq_feedback"])
                else:
                    st.error(st.session_state[f"{tutor_state_key_base}_mcq_feedback"])


    with tab3: # Write & Grade Tab
        st.markdown("#### Write & Grade")
        # ... (Write & Grade placeholder logic as provided previously) ...
        if f"{tutor_state_key_base}_write_grade_task" not in st.session_state:
            st.session_state[f"{tutor_state_key_base}_write_grade_task"] = f"Write a sentence using the word/phrase: '{html.escape(card_data['word_phrase'])}'."
            st.session_state[f"{tutor_state_key_base}_write_grade_feedback"] = None

        st.info(st.session_state[f"{tutor_state_key_base}_write_grade_task"]) 
        user_writing = st.text_area("Your Writing:", height=150, key=f"tutor_user_writing_{card_id}")
        
        if st.button("Get Feedback on Writing", key=f"tutor_grade_writing_{card_id}"):
            if user_writing.strip():
                # TODO: LLM call here
                st.session_state[f"{tutor_state_key_base}_write_grade_feedback"] = "LLM will provide detailed feedback, corrections, and suggestions here."
                st.rerun()
            else: st.warning("Please write something to get feedback.")
        
        if st.session_state[f"{tutor_state_key_base}_write_grade_feedback"]:
            st.markdown("**Feedback:**")
            st.info(st.session_state[f"{tutor_state_key_base}_write_grade_feedback"])

def render_study_deck_view(space_id: str):
    # ... (Full implementation as provided previously, using global field_types_config and st.rerun()) ...
    deck_cards = run_query("SELECT id FROM flashcards WHERE card_space_id = ? ORDER BY created_at ASC", (space_id,), fetchall=True)
    if not deck_cards: st.warning("No cards in this space."); st.session_state.current_view_mode = 'list_cards'; st.session_state.studying_deck_space_id = None; st.rerun(); return
    total_cards = len(deck_cards); current_index = st.session_state.get('study_deck_index', 0)
    if current_index >= total_cards: current_index = total_cards -1; st.session_state.study_deck_index = current_index
    if current_index < 0: current_index = 0; st.session_state.study_deck_index = current_index
    current_card_id = deck_cards[current_index]['id']
    active_space_name = st.session_state.get('active_space_name_for_study', "Deck")
    st.subheader(f"📖 Studying: {active_space_name} ({current_index + 1} / {total_cards})")
    card_data = run_query("SELECT id, word_phrase, image_filename FROM flashcards WHERE id = ?", (current_card_id,), fetchone=True)
    if not card_data: st.error("Card data missing in deck!"); return
    front_html_deck = f"<div style='border: 2px solid #4CAF50; padding: 20px; text-align: center; cursor: pointer; min-height: 200px; display:flex; flex-direction:column; justify-content:center; align-items:center; border-radius: 10px;'><h2>{card_data['word_phrase']}</h2></div>"
    back_html_deck_start = f"<div style='border: 1px solid #38B2AC; background-color: #E6FFFA; color: #234E52; padding: 20px; min-height: 200px; border-radius: 10px;'><h4>{card_data['word_phrase']} - Details:</h4>"
    back_html_deck_content = ""
    if st.session_state.get('study_card_flipped', False):
        card_fields = run_query("SELECT field_type, content FROM flashcard_fields WHERE flashcard_id = ? ORDER BY field_type, sort_order ASC", (current_card_id,), fetchall=True)
        for field_key_local, config_local in field_types_config.items(): 
            contents = [f_row for f_row in card_fields if f_row['field_type'] == field_key_local]
            if contents: back_html_deck_content += f"<p style='margin-top: 10px;'><strong>{config_local['label']}:</strong></p><ul style='margin-bottom:10px; padding-left: 20px;'>"; 
            for entry in contents:
                content_display = str(entry['content'])
                if field_key_local == "pairwise_dialogue":
                    try: dialogue_data = json.loads(content_display); q = dialogue_data.get('question', 'N/A'); a = dialogue_data.get('answer', 'N/A'); content_display = f"<b>Q:</b> {q}<br><b>A:</b> {a}"
                    except: content_display = f"<i>Invalid dialogue format:</i> <pre>{content_display}</pre>"
                elif config_local['json_output']:
                    try: content_display = f"<pre style='background-color: #f0f0f0; padding: 5px; border-radius:3px;'>{json.dumps(json.loads(content_display), indent=2)}</pre>"
                    except: content_display = f"<pre style='background-color: #f0f0f0; padding: 5px; border-radius:3px;'>{content_display}</pre>"
                back_html_deck_content += f"<li style='margin-bottom:5px;'>{content_display}</li>"
            if contents: back_html_deck_content += "</ul>"
        st.markdown(back_html_deck_start + back_html_deck_content + "</div>", unsafe_allow_html=True)
    else:
        st.markdown(front_html_deck, unsafe_allow_html=True)
        if card_data['image_filename']: 
            img_full_path = os.path.join(MEDIA_DIR, space_id, card_data['image_filename'])
            if os.path.exists(img_full_path): st.image(img_full_path, width=150)
            else: logger.warning(f"Image file not found for card {current_card_id} in deck study: {img_full_path}")
    col_nav1, col_nav2, col_nav3 = st.columns([1,1,1])
    with col_nav1:
        if st.button("⬅️ Previous", disabled=current_index <= 0, key=f"prev_deck_main_btn_view_render_{space_id}_{current_card_id}", use_container_width=True): st.session_state.study_deck_index -= 1; st.session_state.study_card_flipped = False; st.rerun()
    with col_nav2:
        if st.button("Flip Card 🃏", key=f"flip_deck_main_btn_view_render_{current_card_id}_{space_id}", use_container_width=True): st.session_state.study_card_flipped = not st.session_state.get('study_card_flipped', False); st.rerun()
    with col_nav3:
        if st.button("Next ➡️", disabled=current_index >= total_cards - 1, key=f"next_deck_main_btn_view_render_{space_id}_{current_card_id}", use_container_width=True): st.session_state.study_deck_index += 1; st.session_state.study_card_flipped = False; st.rerun()
    if st.button("Back to Card List", key="deck_study_back_to_list_btn_main_view_render_v2", use_container_width=True): st.session_state.current_view_mode = 'list_cards'; st.session_state.studying_deck_space_id = None; st.session_state.study_deck_index = 0; st.session_state.study_card_flipped = False; st.rerun()


# --- Streamlit Page Config (Should be the very first Streamlit command) ---
st.set_page_config(layout="wide", page_title="LLM Flashcards")

# --- Main UI Flow (Sidebar is rendered first, then main content area) ---
with st.sidebar:
    st.title("🧠 LLM Flashcards")
    st.header("Card Spaces")
    card_spaces = run_query("SELECT id, name FROM card_spaces ORDER BY name ASC", fetchall=True)
    space_options = {cs['name']: cs['id'] for cs in card_spaces}
    space_names_list = ["Select a Space..."] + list(space_options.keys())
    current_active_space_name = None
    if st.session_state.get('active_space_id'):
        active_space_info = run_query("SELECT name FROM card_spaces WHERE id = ?", (st.session_state.active_space_id,), fetchone=True)
        if active_space_info: current_active_space_name = active_space_info['name']
    try: current_selection_index = space_names_list.index(current_active_space_name) if current_active_space_name and current_active_space_name in space_names_list else 0
    except ValueError: current_selection_index = 0
    chosen_space_name = st.selectbox("Active Space", options=space_names_list, index=current_selection_index, key="sb_active_space_selector_main_ui_key_final")
    if chosen_space_name and chosen_space_name != "Select a Space...":
        if chosen_space_name != current_active_space_name: 
            st.session_state.active_space_id = space_options[chosen_space_name]; st.session_state.current_view_mode = 'list_cards'; st.session_state.editing_flashcard_id = None; st.session_state.studying_flashcard_id = None; st.session_state.studying_deck_space_id = None; st.rerun()
    elif chosen_space_name == "Select a Space..." and st.session_state.active_space_id is not None:
        st.session_state.active_space_id = None; st.session_state.current_view_mode = 'list_cards'; st.rerun()
    with st.form("new_space_form_sidebar_main_ui_key_final", clear_on_submit=True):
        new_space_name_input = st.text_input("New Space Name", key="ti_new_space_sidebar_main_ui_key_final")
        if st.form_submit_button("Create Space"):
            if new_space_name_input.strip():
                space_id_new = str(uuid.uuid4()); now = int(time.time()) 
                run_query("INSERT INTO card_spaces (id, name, created_at, last_updated_at) VALUES (?, ?, ?, ?)", (space_id_new, new_space_name_input.strip(), now, now), commit=True)
                st.session_state.active_space_id = space_id_new; st.session_state.current_view_mode = 'list_cards'; st.success(f"Space '{new_space_name_input.strip()}' created!"); st.rerun()
            else: st.warning("Space name cannot be empty.")
    st.markdown("---"); st.header("Settings")
    current_use_gemini_val_ui = st.session_state.get('use_gemini', False)
    new_use_gemini_val_ui = st.toggle("Use Gemini API", value=current_use_gemini_val_ui, key="toggle_use_gemini_main_setting_ui_key_final")
    if new_use_gemini_val_ui != current_use_gemini_val_ui:
        st.session_state.use_gemini = new_use_gemini_val_ui
        if new_use_gemini_val_ui and (not st.session_state.get('gemini_api_key') or not st.session_state.gemini_api_key.strip()): st.warning("Gemini enabled, but API key is missing below.")
        initialize_llm_processors(); st.rerun() 
    if st.session_state.get('use_gemini', False):
        current_key_val_ui = st.session_state.get('gemini_api_key', "")
        new_key_input_val_ui = st.text_input("Gemini API Key", value=current_key_val_ui, type="password", key="ti_gemini_api_key_setting_ui_key_final")
        if st.button("Save Gemini Key & Re-init", key="btn_save_gemini_key_setting_ui_key_final"):
            if new_key_input_val_ui.strip():
                st.session_state.gemini_api_key = new_key_input_val_ui.strip()
                save_app_settings({"gemini_api_key": st.session_state.gemini_api_key, "ollama_endpoint": st.session_state.ollama_endpoint, "ollama_model": st.session_state.ollama_model})
                initialize_llm_processors() 
                if st.session_state.get('gemini_llm_instance'): st.success("Gemini API Key saved and processor re-initialized!")
                else: st.error("Gemini API Key saved, but processor failed to re-initialize.")
                st.rerun()
            else: st.warning("API Key cannot be empty.")
    with st.expander("Ollama Configuration", expanded=not st.session_state.get('use_gemini', False) ):
        ollama_ep_val_ui = st.text_input("Ollama API Endpoint", value=st.session_state.ollama_endpoint, key="ti_ollama_ep_setting_ui_key_final")
        ollama_mod_val_ui = st.text_input("Ollama Model Name", value=st.session_state.ollama_model, key="ti_ollama_mod_setting_ui_key_final")
        if st.button("Update & Save Ollama Config", key="btn_update_ollama_setting_ui_key_final"):
            st.session_state.ollama_endpoint = ollama_ep_val_ui.strip(); st.session_state.ollama_model = ollama_mod_val_ui.strip()
            save_app_settings({"gemini_api_key": st.session_state.gemini_api_key, "ollama_endpoint": st.session_state.ollama_endpoint, "ollama_model": st.session_state.ollama_model})
            initialize_llm_processors()
            if st.session_state.get('ollama_llm_instance'): st.success("Ollama config updated and processor re-initialized.")
            else: st.error("Ollama config updated, but processor failed to re-initialize.")
            st.rerun()

# Main Content Area Router
if not st.session_state.active_space_id:
    st.info("👈 Select or create a Card Space from the sidebar to begin.")
    if st.session_state.get('current_view_mode') != 'list_cards': 
        st.session_state.current_view_mode = 'list_cards' # Default to list if no space
else:
    # Add logging here to see what current_view_mode is
    logger.info(f"ROUTER: current_view_mode='{st.session_state.current_view_mode}', editing_flashcard_id='{st.session_state.get('editing_flashcard_id')}', studying_flashcard_id='{st.session_state.get('studying_flashcard_id')}', studying_deck_space_id='{st.session_state.get('studying_deck_space_id')}'")

    if st.session_state.current_view_mode == 'edit_card' and st.session_state.editing_flashcard_id:
        render_edit_flashcard_view(st.session_state.editing_flashcard_id, st.session_state.active_space_id)
    
    elif st.session_state.current_view_mode == 'study_card' and st.session_state.studying_flashcard_id:
        # THIS IS WHAT WE WANT TO HIT for your sketch
        render_study_single_card_view(st.session_state.studying_flashcard_id, st.session_state.active_space_id)
    
    elif st.session_state.current_view_mode == 'study_deck' and st.session_state.studying_deck_space_id == st.session_state.active_space_id:
        active_space_info_deck = run_query("SELECT name FROM card_spaces WHERE id = ?", (st.session_state.active_space_id,), fetchone=True)
        st.session_state.active_space_name_for_study = active_space_info_deck['name'] if active_space_info_deck else "Deck"
        render_study_deck_view(st.session_state.active_space_id)
    
    else: # Default to list_cards if no specific view matches or if space is active
        if st.session_state.current_view_mode not in ['list_cards', 'edit_card', 'study_card', 'study_deck']:
             logger.warning(f"Unknown current_view_mode '{st.session_state.current_view_mode}', defaulting to list_cards.")
        st.session_state.current_view_mode = 'list_cards' # Fallback
        render_list_flashcards_view(st.session_state.active_space_id)