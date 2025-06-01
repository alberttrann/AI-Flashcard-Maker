# main_flashcard_app.py
import streamlit as st
import uuid
import os
import io 
import time 
import json 
import asyncio 
import logging 
import re
import inspect 

from db_utils import (
    init_flashcard_db, run_query, load_app_settings, save_app_settings,
    MEDIA_DIR, field_types_config
)
from llm_processors import OllamaLlamaProcessor as ImportedOllamaProcessor
from llm_processors import GeminiAPIProcessor as ImportedGeminiProcessor
from ui_components import (
    render_list_flashcards_view, render_edit_flashcard_view,
    render_study_single_card_view, render_study_deck_view
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'
)

st.set_page_config(layout="wide", page_title="LLM Flashcards")

init_flashcard_db()
app_settings = load_app_settings()

default_session_states = {
    'active_space_id': None,
    'gemini_api_key': app_settings.get("gemini_api_key", ""),
    'ollama_endpoint': app_settings.get("ollama_endpoint", "http://localhost:11434"),
    'ollama_model': app_settings.get("ollama_model", "llama3.1:8b"),
    'current_view_mode': 'list_cards',
    'editing_flashcard_id': None,
    'studying_flashcard_id': None,
    'studying_deck_space_id': None,
    'study_deck_index': 0,
    'study_card_flipped': False,
    'ollama_llm_instance': None,
    'gemini_llm_instance': None,
    'force_gemini_reinit': False,
    'llm_processors_initialized_flag': False
}
for key, value in default_session_states.items():
    if key not in st.session_state: st.session_state[key] = value
if 'use_gemini' not in st.session_state:
    st.session_state.use_gemini = bool(st.session_state.gemini_api_key and st.session_state.gemini_api_key.strip())

def initialize_llm_processors():
    logger.debug("--- Running initialize_llm_processors ---")
    ollama_needs_reinit = (st.session_state.ollama_llm_instance is None or
        getattr(st.session_state.ollama_llm_instance, 'base_url', None) != st.session_state.ollama_endpoint or
        getattr(st.session_state.ollama_llm_instance, 'model_name', None) != st.session_state.ollama_model)
    if ollama_needs_reinit:
        try:
            st.session_state.ollama_llm_instance = ImportedOllamaProcessor(
                base_url=st.session_state.ollama_endpoint, model_name=st.session_state.ollama_model)
            logger.info(f"Ollama processor init/re-init. Type: {type(st.session_state.ollama_llm_instance)}, Class ID: {id(type(st.session_state.ollama_llm_instance))}")
        except Exception as e: logger.error(f"Failed to init Ollama: {e}", exc_info=True); st.session_state.ollama_llm_instance = None

    if st.session_state.gemini_api_key and st.session_state.gemini_api_key.strip():
        gemini_needs_reinit = (st.session_state.gemini_llm_instance is None or st.session_state.force_gemini_reinit)
        if gemini_needs_reinit:
            try:
                st.session_state.gemini_llm_instance = ImportedGeminiProcessor(api_key=st.session_state.gemini_api_key)
                logger.info(f"Gemini processor init/re-init. Type: {type(st.session_state.gemini_llm_instance)}, Class ID: {id(type(st.session_state.gemini_llm_instance))}")
                st.session_state.force_gemini_reinit = False
            except Exception as e: logger.error(f"Failed to init Gemini: {e}", exc_info=True); st.session_state.gemini_llm_instance = None
    elif st.session_state.gemini_llm_instance is not None: 
        logger.info("Gemini API key removed, clearing instance."); st.session_state.gemini_llm_instance = None
    logger.debug("--- Finished initialize_llm_processors ---")

if not st.session_state.llm_processors_initialized_flag:
    initialize_llm_processors()
    st.session_state.llm_processors_initialized_flag = True

def get_active_llm_processor():
    if st.session_state.use_gemini:
        if not st.session_state.gemini_llm_instance and st.session_state.gemini_api_key:
            initialize_llm_processors()
        if not st.session_state.gemini_llm_instance: st.sidebar.warning("Gemini selected, but processor failed.")
        return st.session_state.gemini_llm_instance
    else:
        if not st.session_state.ollama_llm_instance:
            initialize_llm_processors()
        if not st.session_state.ollama_llm_instance: st.sidebar.warning("Ollama selected, but processor failed.")
        return st.session_state.ollama_llm_instance

async def generate_with_active_llm(system_prompt, user_prompt, image_bytes=None, output_json=False):
    active_llm = get_active_llm_processor() 
    if not active_llm:
        logger.error("generate_with_active_llm: LLM Processor is None.")
        return {"error": "LLM Processor not available. Please check settings.", "raw_output": ""}

    instance_class = type(active_llm)
    instance_module_name = instance_class.__module__
    instance_class_name = instance_class.__name__

    logger.info(f"Instance class for LLM call: {instance_class_name}, module: {instance_module_name}, id: {id(instance_class)}")
    logger.info(f"Comparing with ImportedOllamaProcessor: {ImportedOllamaProcessor.__name__} from {ImportedOllamaProcessor.__module__}, id: {id(ImportedOllamaProcessor)}")
    logger.info(f"Comparing with ImportedGeminiProcessor: {ImportedGeminiProcessor.__name__} from {ImportedGeminiProcessor.__module__}, id: {id(ImportedGeminiProcessor)}")
    
    # Type check by module and name string comparison
    is_gemini_by_name = (instance_module_name == ImportedGeminiProcessor.__module__ and \
                         instance_class_name == ImportedGeminiProcessor.__name__)
    is_ollama_by_name = (instance_module_name == ImportedOllamaProcessor.__module__ and \
                         instance_class_name == ImportedOllamaProcessor.__name__)
    
    logger.info(f"Type check by name/module - Is Gemini: {is_gemini_by_name}, Is Ollama: {is_ollama_by_name}")

    try:
        if is_gemini_by_name: # Check by name and module
            logger.info(f"Dispatching to Gemini processor (name/module match) for: {user_prompt[:30]}...")
            return await active_llm.generate_text(system_prompt, user_prompt, image_bytes=image_bytes, output_format_json=output_json, rag_context="")
        elif is_ollama_by_name: # Check by name and module
            logger.info(f"Dispatching to Ollama processor (name/module match) for: {user_prompt[:30]}...")
            prompt_to_send = f"{user_prompt}\n\n[Context: User has also uploaded an image.]" if image_bytes else user_prompt
            return await active_llm.generate_text(system_prompt, prompt_to_send, output_format_json=output_json)
        else: 
            st.error(f"Critical Error: LLM processor type {instance_class_name} from module {instance_module_name} is unknown.")
            logger.error(f"CRITICAL: Unknown LLM processor type after name/module check: {instance_class_name} (Module: {instance_module_name}).")
            return {"error": f"Unknown LLM processor type: {instance_class_name}", "raw_output": ""}
    except Exception as e:
        st.error(f"An unexpected error occurred during the LLM call: {e}")
        logger.error(f"Exception in LLM call dispatch: {e}", exc_info=True)
        return {"error": f"LLM call dispatch error: {e}", "raw_output": ""}

# --- Sidebar UI --- 
with st.sidebar:
    st.title("🧠 LLM Flashcards")
    st.header("Card Spaces")
    card_spaces = run_query("SELECT id, name FROM card_spaces ORDER BY name ASC", fetchall=True)
    space_options = {cs['name']: cs['id'] for cs in card_spaces} if card_spaces else {}
    space_names_list = ["Select a Space..."] + list(space_options.keys())
    
    current_active_space_name = None
    if st.session_state.active_space_id:
        current_active_space_name = next((name for name, id_val in space_options.items() if id_val == st.session_state.active_space_id), None)
    
    current_selection_index = 0
    if current_active_space_name and current_active_space_name in space_names_list:
        current_selection_index = space_names_list.index(current_active_space_name)
    
    chosen_space_name = st.selectbox("Active Space", options=space_names_list, index=current_selection_index, key="sb_active_space_selector_v3")
    
    if chosen_space_name and chosen_space_name != "Select a Space...":
        chosen_space_id = space_options[chosen_space_name]
        if chosen_space_id != st.session_state.active_space_id: 
            st.session_state.active_space_id = chosen_space_id; st.session_state.current_view_mode = 'list_cards' 
            st.session_state.editing_flashcard_id = None; st.session_state.studying_flashcard_id = None
            st.session_state.studying_deck_space_id = None; logger.info(f"Active space changed to: {chosen_space_name} ({chosen_space_id})"); st.rerun()
    elif chosen_space_name == "Select a Space..." and st.session_state.active_space_id is not None:
        st.session_state.active_space_id = None; st.session_state.current_view_mode = 'list_cards'; logger.info("Active space cleared."); st.rerun()

    with st.form("new_space_form_sidebar_v3", clear_on_submit=True):
        new_space_name_input = st.text_input("New Space Name", key="ti_new_space_sidebar_v3")
        if st.form_submit_button("Create Space"):
            if new_space_name_input.strip():
                space_id_new = str(uuid.uuid4()); now = int(time.time()) 
                run_query("INSERT INTO card_spaces (id, name, created_at, last_updated_at) VALUES (?, ?, ?, ?)", (space_id_new, new_space_name_input.strip(), now, now), commit=True)
                st.session_state.active_space_id = space_id_new; st.session_state.current_view_mode = 'list_cards'; st.success(f"Space '{new_space_name_input.strip()}' created!"); logger.info(f"New space created: {new_space_name_input.strip()} ({space_id_new})"); st.rerun()
            else: st.warning("Space name cannot be empty.")
    
    st.markdown("---"); st.header("Settings")
    
    def on_toggle_gemini_changed(): 
        logger.info(f"Gemini toggle callback. New state via st.session_state.use_gemini: {st.session_state.use_gemini}")
        initialize_llm_processors()

    st.toggle("Use Gemini API", value=st.session_state.use_gemini, key="use_gemini", on_change=on_toggle_gemini_changed)
    
    if st.session_state.use_gemini:
        new_key_input_val_ui = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password", key="ti_gemini_api_key_v3")
        if st.button("Save Gemini Key & Re-init", key="btn_save_gemini_key_v3"):
            current_key = st.session_state.gemini_api_key; new_key = new_key_input_val_ui.strip(); key_changed = new_key != current_key
            if not new_key and current_key: 
                st.session_state.gemini_api_key = ""; save_app_settings({"gemini_api_key": "", "ollama_endpoint": st.session_state.ollama_endpoint, "ollama_model": st.session_state.ollama_model})
                st.session_state.force_gemini_reinit = True; initialize_llm_processors(); st.info("Gemini API Key removed."); st.rerun()
            elif new_key and key_changed:
                 st.session_state.gemini_api_key = new_key; save_app_settings({"gemini_api_key": new_key, "ollama_endpoint": st.session_state.ollama_endpoint, "ollama_model": st.session_state.ollama_model})
                 st.session_state.force_gemini_reinit = True; initialize_llm_processors()
                 if st.session_state.gemini_llm_instance: st.success("Gemini API Key updated & processor re-initialized!")
                 else: st.error("Gemini API Key updated, but processor failed to re-initialize.")
                 st.rerun()
            elif new_key and not key_changed and not st.session_state.gemini_llm_instance:
                 st.session_state.force_gemini_reinit = True; initialize_llm_processors()
                 if st.session_state.gemini_llm_instance: st.success("Gemini processor re-initialized!")
                 else: st.error("Gemini processor failed to re-initialize with current key.")
                 st.rerun()
            elif not new_key : st.warning("API Key cannot be empty to save if enabling Gemini.")
            else: st.info("Gemini API Key is already set.")
            
    with st.expander("Ollama Configuration", expanded=not st.session_state.use_gemini ):
        ollama_ep_val_ui = st.text_input("Ollama API Endpoint", value=st.session_state.ollama_endpoint, key="ti_ollama_ep_v3")
        ollama_mod_val_ui = st.text_input("Ollama Model Name", value=st.session_state.ollama_model, key="ti_ollama_mod_v3")
        if st.button("Update & Save Ollama Config", key="btn_update_ollama_v3"):
            if ollama_ep_val_ui.strip() != st.session_state.ollama_endpoint or ollama_mod_val_ui.strip() != st.session_state.ollama_model:
                st.session_state.ollama_endpoint = ollama_ep_val_ui.strip(); st.session_state.ollama_model = ollama_mod_val_ui.strip()
                save_app_settings({"gemini_api_key": st.session_state.gemini_api_key, "ollama_endpoint": st.session_state.ollama_endpoint, "ollama_model": st.session_state.ollama_model})
                initialize_llm_processors(); st.success("Ollama config updated."); st.rerun()
            else: st.info("Ollama configuration unchanged.")

# --- Main Content Area Router ---
if not st.session_state.active_space_id:
    st.info("👈 Select or create a Card Space from the sidebar to begin.")
else:
    logger.debug(f"ROUTER: mode='{st.session_state.current_view_mode}', edit_id='{st.session_state.editing_flashcard_id}', study_id='{st.session_state.studying_flashcard_id}', deck_id='{st.session_state.studying_deck_space_id}'")
    if st.session_state.current_view_mode == 'edit_card' and st.session_state.editing_flashcard_id:
        render_edit_flashcard_view(st.session_state.editing_flashcard_id, st.session_state.active_space_id, generate_with_active_llm)
    elif st.session_state.current_view_mode == 'study_card' and st.session_state.studying_flashcard_id:
        render_study_single_card_view(st.session_state.studying_flashcard_id, st.session_state.active_space_id, generate_with_active_llm)
    elif st.session_state.current_view_mode == 'study_deck' and st.session_state.studying_deck_space_id == st.session_state.active_space_id:
        render_study_deck_view(st.session_state.active_space_id, generate_with_active_llm)
    else: 
        if st.session_state.current_view_mode != 'list_cards':
            logger.info(f"Defaulting to list_cards view from mode: {st.session_state.current_view_mode}")
            st.session_state.current_view_mode = 'list_cards' 
        render_list_flashcards_view(st.session_state.active_space_id)
