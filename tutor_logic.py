# tutor_logic.py
import streamlit as st
import json
import uuid
import time
import html 
from db_utils import run_query

module_logger = st.logger.get_logger(__name__) 

PROMPT_TRICKY_QUESTION = """Given "{WORD_PHRASE}" and context "{DEFINITIONS_CONTEXT}", generate tricky MCQ (JSON: question, options, correct_option_letter, explanation)."""
PROMPT_MCQ_SIMPLE = """For "{WORD_PHRASE}", generate simple MCQ (JSON: question, options, correct_option_letter, explanation)."""
PROMPT_EVALUATE_ANSWER = """Word: "{WORD_PHRASE}", Q: "{QUESTION}", User A: "{USER_ANSWER}". Evaluate (plain text feedback)."""
PROMPT_GRADE_WRITING = """
You are an encouraging and insightful English writing tutor.
The user was asked to: "{TASK_DESCRIPTION}"
The target word/phrase to use was: "{WORD_PHRASE}"

User's Submitted Text:
---
{USER_TEXT}
---

Please provide a single, coherent paragraph of constructive feedback. In your feedback, address the following:
1.  **Usage of "{WORD_PHRASE}"**: Comment on how well it was used. Was it natural? Was it in the correct context? Could a more precise or impactful word have been used if the usage was too ordinary or slightly off?
2.  **Overall Quality**: Briefly touch upon grammar, typos (if any major ones), clarity, and coherence of the text.
3.  **Concrete Suggestions**: Offer one or two specific examples of how the user's sentences could be improved or rephrased for better impact or correctness, especially focusing on the use of "{WORD_PHRASE}".

Keep the tone positive and aim to help the learner improve.
**IMPORTANT: Output your entire response as a single block of plain text. Do NOT use any JSON formatting, markdown lists (unless naturally part of the feedback prose), or any other structured data format.**
"""
PROMPT_SEMANTIC_SYNONYM = """For "{WORD_PHRASE}", generate ONE semantic synonym (JSON: term, explanation, example)."""
PROMPT_SEMANTIC_ANTONYM = """For "{WORD_PHRASE}", generate ONE semantic antonym (JSON: term, explanation, example)."""

def save_tutor_interaction_db(flashcard_id, session_type, llm_generated_content, user_response, llm_feedback):
    interaction_id = str(uuid.uuid4()); created_at = int(time.time())
    llm_generated_content_str = json.dumps(llm_generated_content) if isinstance(llm_generated_content, dict) else str(llm_generated_content)
    llm_feedback_str = json.dumps(llm_feedback) if isinstance(llm_feedback, dict) else str(llm_feedback)
    user_response_str = str(user_response) if user_response is not None else None
    run_query("INSERT INTO tutor_interactions (id, flashcard_id, tutor_session_type, llm_generated_content, user_response, llm_feedback, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (interaction_id, flashcard_id, session_type, llm_generated_content_str, user_response_str, llm_feedback_str, created_at), commit=True)
    module_logger.info(f"Saved tutor interaction {interaction_id} for card {flashcard_id}, type {session_type}")
    return interaction_id

async def _call_llm_for_tutor(prompt, generate_with_llm_func, output_json, feature_name):
    # ... (LLM call as before) ...
    response = await generate_with_llm_func(
        system_prompt="You are a helpful English learning tutor and quiz designer. Provide responses ONLY in the format requested.",
        user_prompt=prompt,
        output_json=output_json
    )

    if isinstance(response, dict) and "error" in response:
        module_logger.error(f"LLM Processor Error for {feature_name}: {response.get('error')}. Raw: {response.get('raw_output', '')[:200]}")
        return response

    if output_json:
        actual_response_data = response
        if isinstance(response, list):
            if len(response) > 0 and isinstance(response[0], dict): # Check if list is not empty
                if len(response) > 1:
                    module_logger.warning(f"LLM for {feature_name} returned a list with {len(response)} dicts; using only the first one.")
                actual_response_data = response[0] 
            else: # List is empty or first item is not a dict
                err_msg = f"LLM returned an list but it's empty or first item not a dict for {feature_name}."
                module_logger.error(f"{err_msg} Content: {str(response)[:300]}")
                return {"error": err_msg, "raw_output": str(response)}
        # ... (rest of the checks: if not isinstance(actual_response_data, dict), etc.)
        if not isinstance(actual_response_data, dict): # After potentially extracting from list
            err_msg = f"LLM did not return a JSON object for {feature_name} (got {type(actual_response_data)} after list check)."
            module_logger.error(f"{err_msg} Content: {str(actual_response_data)[:300]}")
            return {"error": err_msg, "raw_output": str(actual_response_data)}
        return actual_response_data
    # ... (rest of non-JSON handling) ...
    if not response: 
         module_logger.warning(f"LLM returned empty text response for {feature_name}.")
         return "LLM returned no content." 
    if isinstance(response, dict): # Should have been caught by "error" in response if it's an error dict
        module_logger.error(f"Expected plain text for {feature_name} but received dict: {response}")
        return f"Error: Expected plain text, got structured data."
    return str(response)

# --- Individual tutor feature functions call the helper ---
async def generate_tricky_question_llm(word_phrase, definitions_context, generate_with_llm_func):
    prompt = PROMPT_TRICKY_QUESTION.replace("{WORD_PHRASE}", word_phrase).replace("{DEFINITIONS_CONTEXT}", definitions_context)
    return await _call_llm_for_tutor(prompt, generate_with_llm_func, True, "Tricky Question")

async def generate_mcq_llm(word_phrase, generate_with_llm_func):
    prompt = PROMPT_MCQ_SIMPLE.replace("{WORD_PHRASE}", word_phrase)
    return await _call_llm_for_tutor(prompt, generate_with_llm_func, True, "MCQ")

async def evaluate_user_answer_llm(word_phrase, question, user_answer, generate_with_llm_func):
    prompt = PROMPT_EVALUATE_ANSWER.replace("{WORD_PHRASE}", word_phrase).replace("{QUESTION}", question).replace("{USER_ANSWER}", user_answer)
    return await _call_llm_for_tutor(prompt, generate_with_llm_func, False, "Evaluate Answer")

async def grade_writing_llm(word_phrase, task_description, user_text, generate_with_llm_func):
    prompt = PROMPT_GRADE_WRITING.replace("{WORD_PHRASE}", word_phrase).replace("{TASK_DESCRIPTION}", task_description).replace("{USER_TEXT}", user_text)
    # Expecting plain text feedback now, so output_json=False
    return await _call_llm_for_tutor(prompt, generate_with_llm_func, False, "Provide Writing Feedback") 

async def generate_semantic_synonym_llm(word_phrase, generate_with_llm_func):
    prompt = PROMPT_SEMANTIC_SYNONYM.replace("{WORD_PHRASE}", word_phrase)
    return await _call_llm_for_tutor(prompt, generate_with_llm_func, True, "Semantic Synonym")

async def generate_semantic_antonym_llm(word_phrase, generate_with_llm_func):
    prompt = PROMPT_SEMANTIC_ANTONYM.replace("{WORD_PHRASE}", word_phrase)
    return await _call_llm_for_tutor(prompt, generate_with_llm_func, True, "Semantic Antonym")

def get_past_tutor_interactions(flashcard_id, session_type=None):
    # ... (this function remains the same as the last good version) ...
    query = "SELECT id, tutor_session_type, llm_generated_content, user_response, llm_feedback, created_at FROM tutor_interactions WHERE flashcard_id = ?"
    params = [flashcard_id]
    if session_type: query += " AND tutor_session_type = ?"; params.append(session_type)
    query += " ORDER BY created_at DESC"
    interactions = run_query(query, tuple(params), fetchall=True)
    parsed_interactions = []
    if interactions:
        for row_data in interactions:
            interaction = dict(row_data)
            for key in ['llm_generated_content', 'llm_feedback']:
                if interaction.get(key) and isinstance(interaction[key], str) and interaction[key].strip().startswith(("{", "[")):
                    try: interaction[key] = json.loads(interaction[key])
                    except json.JSONDecodeError: module_logger.warning(f"Could not parse JSON for field {key}, interaction {interaction.get('id')}")
            parsed_interactions.append(interaction)
    return parsed_interactions
