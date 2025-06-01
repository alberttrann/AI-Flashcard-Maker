import streamlit as st
import os
import json
import html
import time
import uuid
import asyncio 

from db_utils import run_query, MEDIA_DIR, field_types_config 
import tutor_logic 

module_logger = st.logger.get_logger(__name__) 

def render_live_tutor_tabs(card_data, current_card_id, tutor_state_key_base, generate_with_llm_func_ref):
    word_phrase_for_tutor = card_data['word_phrase']
    tab_labels = ["🎯 Tricky Questions", "📝 MCQs", "✍️ Write & Grade", "🔄 Synonyms/Antonyms"]
    tab_tricky, tab_mcq, tab_write, tab_syn_ant = st.tabs(tab_labels)

    with tab_tricky:
        st.markdown("#### Practice with Tricky Questions")
        past_tricky_q = tutor_logic.get_past_tutor_interactions(current_card_id, "tricky_question")
        if past_tricky_q:
            with st.expander("Past Tricky Questions & Answers", expanded=False):
                for i, interaction in enumerate(past_tricky_q):
                    q_data = interaction.get('llm_generated_content', {})
                    user_a = html.escape(str(interaction.get('user_response', 'N/A')))
                    feedback = str(interaction.get('llm_feedback', 'N/A')) # Feedback might be complex HTML
                    st.markdown(f"**Q{i+1}:** {html.escape(q_data.get('question', 'N/A'))}")
                    if "options" in q_data and isinstance(q_data.get("options"),list): st.caption(f"Options were: {', '.join(map(str,q_data['options']))}")
                    st.caption(f"Your answer: {user_a}")
                    # Display feedback carefully, it might contain HTML from our formatting
                    if "Correct!" in feedback: st.success(f"Feedback: {feedback}", icon="🎉") 
                    else: st.warning(f"Feedback: {feedback}")
                    st.divider()

        q_data_key = f"{tutor_state_key_base}_tricky_q_data"
        user_answer_key = f"{tutor_state_key_base}_tricky_q_user_answer" # For open-ended
        feedback_key = f"{tutor_state_key_base}_tricky_q_feedback"
        mcq_choice_key = f"{tutor_state_key_base}_tricky_q_mcq_choice" # For MCQ selection from options list

        if q_data_key not in st.session_state: st.session_state[q_data_key] = None
        if user_answer_key not in st.session_state: st.session_state[user_answer_key] = ""
        if feedback_key not in st.session_state: st.session_state[feedback_key] = None
        if mcq_choice_key not in st.session_state: st.session_state[mcq_choice_key] = None # Will store the selected option value
        
        if st.button("Get a Tricky Question", key=f"tutor_get_tricky_q_{current_card_id}"):
            with st.spinner("Generating tricky question..."):
                definitions_field = run_query("SELECT content FROM flashcard_fields WHERE flashcard_id = ? AND field_type = 'definition'", (current_card_id,), fetchall=True)
                definitions_context = "\n".join([html.escape(d['content']) for d in definitions_field]) if definitions_field else "No definition context available."
                
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                q_data_result = loop.run_until_complete(tutor_logic.generate_tricky_question_llm(word_phrase_for_tutor, definitions_context, generate_with_llm_func_ref))
                loop.close()
                if q_data_result and isinstance(q_data_result, dict) and "error" not in q_data_result:
                    st.session_state[q_data_key] = q_data_result
                    st.session_state[user_answer_key] = ""
                    st.session_state[feedback_key] = None
                    st.session_state[mcq_choice_key] = None 
                    tutor_logic.save_tutor_interaction_db(current_card_id, "tricky_question", q_data_result, None, None)
                else: st.error(f"Could not generate tricky question: {q_data_result.get('error', 'Unknown error') if isinstance(q_data_result, dict) else 'LLM call failed.'}")
            st.rerun()

# ui_components.py -> render_live_tutor_tabs -> with tab_tricky:

        current_q_data = st.session_state.get(q_data_key)
        if current_q_data:
            st.info(f"**Question:** {html.escape(current_q_data.get('question', 'Error: Question not found in data.'))}")
            
            # --- MODIFIED OPTIONS HANDLING ---
            options_for_radio = []
            options_source = current_q_data.get("options")

            if isinstance(options_source, list): # LLM returned a list of strings (ideal)
                options_for_radio = options_source
            elif isinstance(options_source, dict): # LLM returned a dict {'a': val_a, 'b': val_b}
                module_logger.info("Tricky question options received as dict, converting to list for radio.")
                # We need to ensure a consistent order, e.g., by sorting keys or assuming 'a','b','c','d'
                # For simplicity, let's assume keys 'a', 'b', 'c', 'd' if they exist, or sort other keys.
                # A more robust way is to ensure LLM returns list or a dict with ordered keys we expect.
                # For now, if it's a dict, we'll try to get values in a,b,c,d order if present, else sorted keys.
                if all(k in options_source for k in ['a', 'b', 'c', 'd']):
                    options_for_radio = [
                        options_source['a'], 
                        options_source['b'], 
                        options_source['c'], 
                        options_source['d']
                    ]
                else: # Fallback to sorted keys if 'a','b','c','d' are not all present
                    for key in sorted(options_source.keys()):
                        options_for_radio.append(options_source[key])
            else:
                module_logger.error(f"Tricky question options are neither list nor dict: {type(options_source)}")

            if options_for_radio: # Only proceed if we have options to display
                module_logger.debug(f"Tricky Question MCQ Options for radio (processed): {options_for_radio}")
                
                current_selection_value = st.session_state.get(mcq_choice_key)
                idx_for_radio = None
                if current_selection_value is not None and current_selection_value in options_for_radio:
                    try:
                        idx_for_radio = options_for_radio.index(current_selection_value)
                    except ValueError: 
                         module_logger.warning(f"Previous selection '{current_selection_value}' not in new options '{options_for_radio}'.")
                         st.session_state[mcq_choice_key] = None 

                selected_option_value = st.radio(
                    "Your choice:", 
                    options_for_radio, # Use the processed list
                    key=f"tricky_q_radio_{current_card_id}_{current_q_data.get('question','')[:10]}", 
                    index=idx_for_radio,
                    format_func=lambda x: html.escape(str(x)) 
                )
                st.session_state[mcq_choice_key] = selected_option_value

                if st.button("Check MCQ Answer", key=f"tutor_check_tricky_mcq_{current_card_id}"):
                    chosen_option_from_state = st.session_state.get(mcq_choice_key)
                    if chosen_option_from_state is not None:
                        correct_letter = current_q_data.get("correct_option_letter", "").upper()
                        
                        # Determine the correct index based on the original options_for_radio list
                        # and the letter provided by the LLM.
                        chosen_idx = -1
                        try:
                            chosen_idx = options_for_radio.index(chosen_option_from_state)
                        except ValueError:
                            module_logger.error(f"Chosen option '{chosen_option_from_state}' not found in options_for_radio: {options_for_radio}")
                            st.session_state[feedback_key] = "Error: Your selection was not recognized. Please try again."
                            st.rerun(); return

                        correct_idx_from_letter = -1
                        if 'A' <= correct_letter <= 'D': # Assuming up to 4 options
                           correct_idx_from_letter = ord(correct_letter) - ord('A')
                        elif correct_letter.isdigit() and 0 <= int(correct_letter) < len(options_for_radio): # If LLM gives 0-indexed number
                           correct_idx_from_letter = int(correct_letter)


                        is_correct = (chosen_idx == correct_idx_from_letter)
                        explanation = html.escape(current_q_data.get('explanation', 'No explanation provided.'))
                        feedback_text = f"**Explanation:** {explanation}"
                        if is_correct: final_feedback = f"Correct! 🎉\n{feedback_text}"
                        else: 
                            correct_ans_text = "N/A"
                            if 0 <= correct_idx_from_letter < len(options_for_radio):
                                correct_ans_text = options_for_radio[correct_idx_from_letter]
                            final_feedback = f"Not quite. The best answer was: **{html.escape(str(correct_ans_text))}**.\n{feedback_text}"
                        
                        st.session_state[feedback_key] = final_feedback
                        tutor_logic.save_tutor_interaction_db(current_card_id, "tricky_question", current_q_data, str(chosen_option_from_state), final_feedback)
                        st.rerun()
                    else: st.warning("Please select an option.")
            # ... (rest of open-ended question logic) ...
            else: # No valid options found for MCQ
                module_logger.warning(f"No options to display for Tricky Question MCQ. Data: {current_q_data}")
                st.caption("Could not load options for this question.")
                # (open-ended question logic would follow here if that's the fallback)
                st.session_state[user_answer_key] = st.text_area("Your Answer (if open-ended):", value=st.session_state[user_answer_key], key=f"tutor_user_answer_tricky_open_{current_card_id}", height=100)
                # ... (rest of open-ended check button logic) ...

        feedback_to_display = st.session_state.get(feedback_key)
        if feedback_to_display:
            if "Correct!" in feedback_to_display or "Excellent!" in feedback_to_display : st.success(feedback_to_display, icon="✅")
            else: st.warning(feedback_to_display)
    
    with tab_mcq:
        st.markdown("#### Practice with MCQs")
        past_mcqs = tutor_logic.get_past_tutor_interactions(current_card_id, "mcq")
        if past_mcqs:
            # ... (display past MCQs logic - seems okay)
            with st.expander("Past MCQs", expanded=False):
                for i, interaction in enumerate(past_mcqs):
                    mcq_data = interaction.get('llm_generated_content', {})
                    st.markdown(f"**Q{i+1}:** {html.escape(mcq_data.get('question', 'N/A'))}")
                    st.write(f"Options: {', '.join(map(lambda x: html.escape(str(x)), mcq_data.get('options',[])))}")
                    st.caption(f"Your choice: {html.escape(str(interaction.get('user_response', 'N/A')))}")
                    feedback_str = str(interaction.get('llm_feedback', 'N/A'))
                    if "Correct!" in feedback_str: st.success(f"Feedback: {feedback_str}", icon="🎉")
                    else: st.error(f"Feedback: {feedback_str}")
                    st.divider()


        mcq_data_key = f"{tutor_state_key_base}_mcq_data"
        mcq_choice_actual_value_key = f"{tutor_state_key_base}_mcq_user_choice_value" 
        mcq_feedback_key = f"{tutor_state_key_base}_mcq_feedback"

        if mcq_data_key not in st.session_state: st.session_state[mcq_data_key] = None
        if mcq_choice_actual_value_key not in st.session_state: st.session_state[mcq_choice_actual_value_key] = None
        if mcq_feedback_key not in st.session_state: st.session_state[mcq_feedback_key] = None
        
        if st.button("Generate MCQ", key=f"tutor_get_mcq_{current_card_id}"):
            with st.spinner("Generating MCQ..."):
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                mcq_data_result = loop.run_until_complete(tutor_logic.generate_mcq_llm(word_phrase_for_tutor, generate_with_llm_func_ref))
                loop.close()
                if mcq_data_result and isinstance(mcq_data_result, dict) and "error" not in mcq_data_result:
                    st.session_state[mcq_data_key] = mcq_data_result
                    st.session_state[mcq_choice_actual_value_key] = None 
                    st.session_state[mcq_feedback_key] = None
                    tutor_logic.save_tutor_interaction_db(current_card_id, "mcq", mcq_data_result, None, None)
                else: st.error(f"Could not generate MCQ: {mcq_data_result.get('error', 'Unknown error') if isinstance(mcq_data_result, dict) else 'LLM call failed.'}")
            st.rerun()

        current_mcq_data = st.session_state.get(mcq_data_key)
        if current_mcq_data:
            st.write(f"**Question:** {html.escape(current_mcq_data.get('question', 'Error: Question missing'))}")
            options_source_mcq = current_mcq_data.get("options") 
            
            module_logger.debug(f"Simple MCQ Options source type: {type(options_source_mcq)}, value: {options_source_mcq}")

            options_for_radio_mcq = [] # Define it before the if/elif/else
            if isinstance(options_source_mcq, list):
                options_for_radio_mcq = [str(opt) for opt in options_source_mcq if str(opt).strip()] # Ensure all are strings and non-empty
            elif isinstance(options_source_mcq, dict):
                module_logger.info("Simple MCQ options received as dict, converting to list for radio.")
                temp_options_list = []
                # Try specific keys first for consistent ordering if LLM uses them
                for key_char in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']: # Check common MCQ keys
                    if key_char in options_source_mcq:
                        temp_options_list.append(str(options_source_mcq[key_char]))
                
                if not temp_options_list: # Fallback if specific keys not found, use sorted keys from dict
                    for key in sorted(options_source_mcq.keys()):
                        temp_options_list.append(str(options_source_mcq[key]))
                options_for_radio_mcq = [opt for opt in temp_options_list if opt.strip()]


            if not options_for_radio_mcq: 
                 module_logger.error(f"No valid options could be extracted for Simple MCQ. Original options data: {options_source_mcq}")
                 st.caption("Error: Options for MCQ could not be displayed.")
            else: # Only display radio if options_for_radio_mcq is populated
                current_selection_value_mcq = st.session_state.get(mcq_choice_actual_value_key)
                idx_for_radio_mcq = None
                if current_selection_value_mcq is not None and current_selection_value_mcq in options_for_radio_mcq:
                    try: idx_for_radio_mcq = options_for_radio_mcq.index(current_selection_value_mcq)
                    except ValueError: st.session_state[mcq_choice_actual_value_key] = None 

                selected_option_value_mcq = st.radio(
                    "Your answer:", 
                    options_for_radio_mcq, # CORRECTED: Use the processed list
                    format_func=lambda x: html.escape(x), # x is already string from options_for_radio_mcq
                    key=f"tutor_mcq_radio_{current_card_id}_{current_mcq_data.get('question', '')[:10]}", 
                    index=idx_for_radio_mcq
                )
                st.session_state[mcq_choice_actual_value_key] = selected_option_value_mcq

                if st.button("Check MCQ Answer", key=f"tutor_check_mcq_{current_card_id}"):
                    chosen_option_from_state_mcq = st.session_state.get(mcq_choice_actual_value_key)
                    if chosen_option_from_state_mcq is not None:
                        correct_letter = current_mcq_data.get("correct_option_letter", "").upper()
                        
                        chosen_idx_mcq = -1
                        try: 
                            chosen_idx_mcq = options_for_radio_mcq.index(chosen_option_from_state_mcq) # CORRECTED
                        except ValueError: 
                            module_logger.error(f"Chosen option '{chosen_option_from_state_mcq}' not in options_for_radio_mcq during check: {options_for_radio_mcq}")
                            st.session_state[mcq_feedback_key] = "Error: Selection error."; st.rerun(); return
                        
                        correct_idx_from_letter = ord(correct_letter) - ord('A') if 'A' <= correct_letter <= 'D' else -1
                        is_correct_mcq = (chosen_idx_mcq == correct_idx_from_letter)
                        explanation_mcq = html.escape(current_mcq_data.get('explanation', 'No explanation provided.'))
                        feedback_text_mcq = f"**Explanation:** {explanation_mcq}"
                        
                        if is_correct_mcq: 
                            final_feedback_mcq = f"Correct! 🎉\n{feedback_text_mcq}"
                        else: 
                            correct_ans_text_mcq = "N/A"
                            if 0 <= correct_idx_from_letter < len(options_for_radio_mcq): # CORRECTED
                                correct_ans_text_mcq = options_for_radio_mcq[correct_idx_from_letter] # CORRECTED
                            final_feedback_mcq = f"Not quite. The correct answer was: **{html.escape(correct_ans_text_mcq)}**.\n{feedback_text_mcq}" # html.escape for safety
                        
                        st.session_state[mcq_feedback_key] = final_feedback_mcq
                        tutor_logic.save_tutor_interaction_db(current_card_id, "mcq", current_mcq_data, str(chosen_option_from_state_mcq), final_feedback_mcq)
                        st.rerun()
            
            feedback_mcq_display = st.session_state.get(mcq_feedback_key)
            if feedback_mcq_display:
                if "Correct!" in feedback_mcq_display: st.success(feedback_mcq_display, icon="✅")
                else: st.error(feedback_mcq_display)
    
    # ... (tab_write and tab_syn_ant remain the same as the "Aiming for final version" code)
    with tab_write:
        st.markdown("#### Write & Grade")
        
        # Display past writing submissions and feedback
        past_writings = tutor_logic.get_past_tutor_interactions(current_card_id, "write_grade") # Changed type from "write_grade"
        if past_writings:
            with st.expander("Past Writing Submissions & Feedback", expanded=False):
                for i, interaction in enumerate(past_writings):
                    task_description_past = str(interaction.get('llm_generated_content', "N/A")) # Task was stored here
                    user_text_past = str(interaction.get('user_response', 'N/A'))
                    feedback_text_past = str(interaction.get('llm_feedback', 'N/A')) # Feedback is now plain text

                    st.markdown(f"**Submission {i+1} for task:**")
                    st.caption(html.escape(task_description_past))
                    st.markdown("**Your text:**")
                    st.text_area(f"Past Writing {i+1}", value=user_text_past, height=100, disabled=True, key=f"past_write_text_{interaction.get('id')}_{i}")
                    st.markdown("**Feedback received:**")
                    st.info(feedback_text_past) # Display feedback as a paragraph in an info box
                    st.divider()

        task_desc_key = f"{tutor_state_key_base}_write_grade_task_desc"
        user_writing_key = f"{tutor_state_key_base}_user_writing"
        feedback_key = f"{tutor_state_key_base}_write_grade_feedback_text" # New key for plain text feedback

        # Initialize session state for this tab if not present
        if task_desc_key not in st.session_state:
            st.session_state[task_desc_key] = f"Write a sentence or short paragraph using the word/phrase: '{html.escape(word_phrase_for_tutor)}'."
        if user_writing_key not in st.session_state:
            st.session_state[user_writing_key] = ""
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None
        
        st.info(st.session_state[task_desc_key]) # Display the current task
        
        st.session_state[user_writing_key] = st.text_area(
            "Your Writing:", 
            value=st.session_state[user_writing_key], 
            height=150, 
            key=f"tutor_user_writing_{current_card_id}"
        )
        
        if st.button("Get Feedback on Writing", key=f"tutor_grade_writing_{current_card_id}"):
            if st.session_state[user_writing_key].strip():
                with st.spinner("Getting feedback on your writing..."):
                    loop = None
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        feedback_text_result = loop.run_until_complete(
                            tutor_logic.grade_writing_llm(
                                word_phrase_for_tutor, 
                                st.session_state[task_desc_key], 
                                st.session_state[user_writing_key], 
                                generate_with_llm_func_ref
                            )
                        )
                    except Exception as e_gw_call:
                        module_logger.error(f"Error during grade_writing LLM call: {e_gw_call}", exc_info=True)
                        st.error(f"Failed to get writing feedback: {e_gw_call}")
                        feedback_text_result = {"error": f"Writing feedback call failed: {e_gw_call}"}
                    finally:
                        if loop and not loop.is_closed():
                            loop.close()
                
                # Check if the result is an error dictionary (from _call_llm_for_tutor or processor)
                if isinstance(feedback_text_result, dict) and "error" in feedback_text_result:
                    st.error(f"Could not get feedback: {feedback_text_result.get('error')}. Raw: {feedback_text_result.get('raw_output','')[:100]}")
                    st.session_state[feedback_key] = None # Clear previous good feedback
                elif feedback_text_result: # Should be a string
                    st.session_state[feedback_key] = str(feedback_text_result)
                    # Save interaction: task description as llm_generated_content, user text, and LLM feedback text
                    tutor_logic.save_tutor_interaction_db(
                        current_card_id, 
                        "write_grade", # Consistent session type
                        st.session_state[task_desc_key], 
                        st.session_state[user_writing_key], 
                        str(feedback_text_result)
                    )
                else:
                    st.error("Received no feedback from the LLM.")
                    st.session_state[feedback_key] = None
                st.rerun()
            else:
                st.warning("Please write something to get feedback.")
        
        # Display current feedback (now expected to be a plain text paragraph)
        current_feedback_text = st.session_state.get(feedback_key)
        if current_feedback_text:
            st.markdown("---") # Separator
            st.markdown("#### 📝 Feedback on Your Writing:")
            st.info(current_feedback_text) # Display as a paragraph in an info box

    with tab_syn_ant:
        st.markdown("#### Semantic Synonyms & Antonyms")
        # ... (display past interactions logic remains the same)
        past_syn = tutor_logic.get_past_tutor_interactions(current_card_id, "synonym")
        past_ant = tutor_logic.get_past_tutor_interactions(current_card_id, "antonym")

        col_syn_disp, col_ant_disp = st.columns(2)
        with col_syn_disp:
            st.markdown("**Generated Synonyms**")
            # ... (display past_syn)
            if past_syn:
                for i, interaction in enumerate(past_syn):
                    item_data = interaction.get('llm_generated_content', {})
                    with st.container(border=True, key=f"syn_container_{interaction.get('id')}_{i}"):
                        st.markdown(f"**{i+1}. {html.escape(item_data.get('term', 'N/A'))}**")
                        st.caption(f"_{html.escape(item_data.get('explanation', 'N/A'))}_")
                        st.markdown(f"> {html.escape(item_data.get('example', 'N/A'))}")
            else: st.caption("No synonyms generated yet.")


            if st.button(f"Generate Synonym", key=f"tutor_gen_syn_{current_card_id}"):
                loop = None # Initialize loop to None
                with st.spinner("Generating synonym..."):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        syn_data_result = loop.run_until_complete(
                            tutor_logic.generate_semantic_synonym_llm(word_phrase_for_tutor, generate_with_llm_func_ref)
                        )
                    except Exception as e_syn_call:
                        module_logger.error(f"Error during synonym LLM call setup/execution: {e_syn_call}", exc_info=True)
                        st.error(f"Failed to initiate synonym generation: {e_syn_call}")
                        syn_data_result = {"error": f"Synonym generation call failed: {e_syn_call}"} # Create error dict
                    finally:
                        if loop and not loop.is_closed():
                            loop.close()
                            module_logger.debug("Synonym generation event loop closed.")
                
                # Process result after spinner and loop handling
                if syn_data_result and isinstance(syn_data_result, dict) and "error" not in syn_data_result:
                    if all(k in syn_data_result for k in ["term", "explanation", "example"]):
                        tutor_logic.save_tutor_interaction_db(current_card_id, "synonym", syn_data_result, None, None)
                    else:
                        module_logger.error(f"Synonym data from LLM missing expected keys: {syn_data_result}")
                        st.error(f"LLM returned incomplete data for synonym. Raw: {str(syn_data_result)[:200]}")
                elif syn_data_result and isinstance(syn_data_result, dict) and "error" in syn_data_result:
                    st.error(f"Could not generate synonym: {syn_data_result.get('error')} Raw: {syn_data_result.get('raw_output','')[:100]}")
                else: 
                    st.error(f"Could not generate synonym (unexpected response type: {type(syn_data_result)}).")
                st.rerun()
        with col_ant_disp:
            st.markdown("**Generated Antonyms**")
            # ... (display past_ant)
            if past_ant:
                for i, interaction in enumerate(past_ant):
                    item_data = interaction.get('llm_generated_content', {})
                    with st.container(border=True, key=f"ant_container_{interaction.get('id')}_{i}"):
                        st.markdown(f"**{i+1}. {html.escape(item_data.get('term', 'N/A'))}**")
                        st.caption(f"_{html.escape(item_data.get('explanation', 'N/A'))}_")
                        st.markdown(f"> {html.escape(item_data.get('example', 'N/A'))}")
            else: st.caption("No antonyms generated yet.")

            if st.button(f"Generate Antonym", key=f"tutor_gen_ant_{current_card_id}"):
                loop = None # Initialize loop to None
                with st.spinner("Generating antonym..."):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        ant_data_result = loop.run_until_complete(
                            tutor_logic.generate_semantic_antonym_llm(word_phrase_for_tutor, generate_with_llm_func_ref)
                        )
                    except Exception as e_ant_call:
                        module_logger.error(f"Error during antonym LLM call setup/execution: {e_ant_call}", exc_info=True)
                        st.error(f"Failed to initiate antonym generation: {e_ant_call}")
                        ant_data_result = {"error": f"Antonym generation call failed: {e_ant_call}"} # Create error dict
                    finally:
                        if loop and not loop.is_closed():
                            loop.close()
                            module_logger.debug("Antonym generation event loop closed.")

                # Process result after spinner and loop handling
                if ant_data_result and isinstance(ant_data_result, dict) and "error" not in ant_data_result:
                    if all(k in ant_data_result for k in ["term", "explanation", "example"]):
                        tutor_logic.save_tutor_interaction_db(current_card_id, "antonym", ant_data_result, None, None)
                    else:
                        module_logger.error(f"Antonym data from LLM missing expected keys: {ant_data_result}")
                        st.error(f"LLM returned incomplete data for antonym. Raw: {str(ant_data_result)[:200]}")
                elif ant_data_result and isinstance(ant_data_result, dict) and "error" in ant_data_result:
                    st.error(f"Could not generate antonym: {ant_data_result.get('error')} Raw: {ant_data_result.get('raw_output','')[:100]}")
                else:
                    st.error(f"Could not generate antonym (unexpected response type: {type(ant_data_result)}).")
                st.rerun()

# --- render_study_single_card_view ---
def render_study_single_card_view(card_id: str, space_id: str, generate_with_llm_func_ref):
    module_logger.info(f"Rendering SINGLE CARD view for card_id: {card_id}") # Corrected
    # ... (The rest of render_study_single_card_view as provided in the "Aiming for final version" response, 
    #      making sure it calls render_live_tutor_tabs at the end and uses module_logger for its logs)
    # Copy the full function body from previous correct version, ensuring it uses 'module_logger'
    # and calls render_live_tutor_tabs(card_data, card_id, tutor_state_key_base, generate_with_llm_func_ref)

    card_data = run_query("SELECT id, word_phrase, image_filename FROM flashcards WHERE id = ? AND card_space_id = ?", (card_id, space_id), fetchone=True)
    if not card_data: module_logger.error(f"Flashcard not found for study: {card_id}"); st.error("Flashcard for study not found."); st.session_state.current_view_mode = 'list_cards'; st.session_state.studying_flashcard_id = None; st.rerun(); return

    st.subheader(f"📖 Studying Card: {html.escape(card_data['word_phrase'])}")
    st.markdown("---")

    if 'study_card_flipped' not in st.session_state or st.session_state.studying_flashcard_id != card_id:
        st.session_state.study_card_flipped = False
        st.session_state.studying_flashcard_id = card_id

    col_card_area, col_image_area = st.columns([0.65, 0.35])
    with col_card_area:
        card_container_height = "450px" 
        if not st.session_state.study_card_flipped:
            front_html_style = f"""border: 2px solid #718096; background-color: #FFFFFF; color: #2D3748; padding: 20px; text-align: center; height: {card_container_height}; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 10px; font-size: 2.8em; font-weight: bold; box-shadow: 0 6px 12px rgba(0,0,0,0.15);"""
            st.markdown(f"<div style='{front_html_style}'>{html.escape(card_data['word_phrase'])}</div>", unsafe_allow_html=True)
        else:
            back_html_style = f"""border: 2px solid #38B2AC; background-color: #F7FAFC; color: #2D3748; padding: 20px; height: {card_container_height}; border-radius: 10px; overflow-y: auto; box-shadow: 0 6px 12px rgba(0,0,0,0.15);"""
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
                        content_display = entry_content_str; item_style = "background-color: #EDF2F7; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"
                        if field_key_local == "pairwise_dialogue":
                            try: dialogue_data = json.loads(content_display); q = html.escape(dialogue_data.get('question', 'N/A')); a = html.escape(dialogue_data.get('answer', 'N/A')); content_display = f"<div style='{item_style}'><strong style='color:#3182CE;'>Q:</strong> {q}<br><strong style='color:#3182CE;'>A:</strong> {a}</div>"
                            except Exception as e: content_display = f"<div style='{item_style} background-color: #FFF5F5; color: #C53030;'><i>Invalid dialogue:</i> <pre>{html.escape(content_display)}</pre></div>"; module_logger.error(f"SingleCard: Error displaying pairwise_dialogue: {e}")
                        elif field_key_local == "word_family":
                            try:
                                parsed_json = json.loads(content_display)
                                wf_html = ""
                                for key, value_list in parsed_json.items():
                                    disp_key = html.escape(key.replace('_', ' ')).capitalize()
                                    if isinstance(value_list, list) and value_list and any(str(v).strip() for v in value_list):
                                        wf_html += f"<div style='margin-bottom:3px;'><strong style='color:#007A5E;'>{disp_key}:</strong> {', '.join(html.escape(str(v)) for v in value_list if str(v).strip())}</div>"
                                    elif isinstance(value_list, str) and value_list.strip():
                                        wf_html += f"<div style='margin-bottom:3px;'><strong style='color:#007A5E;'>{disp_key}:</strong> {html.escape(value_list)}</div>"
                                no_terms_span = "<span style='font-style:italic; color: #718096;'>No terms provided.</span>"
                                inner_html_for_wf = wf_html if wf_html else no_terms_span
                                content_display = f"<div style='{item_style} background-color: #E6FFFA;'>{inner_html_for_wf}</div>"
                            except Exception as e_wf:
                                content_display = f"<div style='{item_style} background-color: #FFF5F5; color: #C53030;'><i>Invalid Word Family JSON:</i> <pre>{html.escape(content_display)}</pre></div>"
                                module_logger.error(f"SingleCard: Error displaying word_family: {e_wf}")
                        elif config_local['json_output']: 
                             try: parsed_json = json.loads(content_display); content_display = f"<pre style='{item_style} white-space: pre-wrap; word-wrap: break-word; background-color: #F0F0F0;'>{html.escape(json.dumps(parsed_json, indent=2))}</pre>"
                             except Exception as e: content_display = f"<pre style='{item_style} white-space: pre-wrap; word-wrap: break-word; background-color: #FFF5F5; color: #C53030;'><i>Invalid JSON:</i> {html.escape(content_display)}</pre>"; module_logger.error(f"SingleCard: Error displaying other JSON: {e}")
                        else: content_display = f"<div style='{item_style}'>{html.escape(content_display).replace(chr(10), '<br>')}</div>"
                        back_html_content += f"<li>{content_display}</li>"
                    back_html_content += "</ul>"
            st.markdown(f"<div style='{back_html_style}'>{back_html_content}</div>", unsafe_allow_html=True)
    with col_image_area:
        if card_data['image_filename']:
            img_full_path = os.path.join(MEDIA_DIR, space_id, card_data['image_filename'])
            if os.path.exists(img_full_path): st.image(img_full_path, use_container_width=True, caption="Attached Image")
            else: module_logger.warning(f"Image file not found: {img_full_path}"); st.markdown(f"<div style='height: {card_container_height}; display:flex; align-items:center; justify-content:center; border: 2px dashed #CBD5E0; border-radius:10px; color: #A0AEC0; text-align:center;'>Image not found<br><small>{html.escape(card_data['image_filename'])}</small></div>", unsafe_allow_html=True)
        else: st.markdown(f"<div style='height: {card_container_height}; display:flex; align-items:center; justify-content:center; border: 2px dashed #CBD5E0; border-radius:10px; color: #A0AEC0;'>No Image Attached</div>", unsafe_allow_html=True)

    st.markdown("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Flip Card 🃏", key=f"flip_single_study_btn_view_render_{card_id}", use_container_width=True):
            st.session_state.study_card_flipped = not st.session_state.study_card_flipped; st.rerun()
    with col_b2:
        if st.button("Back to Card List", key=f"study_single_back_to_list_btn_view_render_{card_id}", use_container_width=True):
            st.session_state.current_view_mode = 'list_cards'; st.session_state.studying_flashcard_id = None; st.session_state.study_card_flipped = False; st.rerun()
    st.markdown("---")
    
    st.subheader("🎓 Live Tutor")
    tutor_state_key_base = f"tutor_state_{card_id}" 
    render_live_tutor_tabs(card_data, card_id, tutor_state_key_base, generate_with_llm_func_ref)


# --- render_study_deck_view ---
def render_study_deck_view(space_id: str, generate_with_llm_func_ref):
    module_logger.info(f"Rendering DECK view for space_id: {space_id}") # Corrected
    # ... (The rest of render_study_deck_view as provided in the "Aiming for final version" response)
    # Ensure all its internal logger calls also use module_logger
    # and it calls render_live_tutor_tabs(card_data, current_card_id, tutor_state_key_base, generate_with_llm_func_ref)
    deck_cards_ids = run_query("SELECT id FROM flashcards WHERE card_space_id = ? ORDER BY created_at ASC", (space_id,), fetchall=True)
    if not deck_cards_ids: module_logger.warning(f"No cards in deck for space {space_id}"); st.warning("No cards in this space to study as a deck."); st.session_state.current_view_mode = 'list_cards'; st.session_state.studying_deck_space_id = None; st.rerun(); return
    total_cards = len(deck_cards_ids)
    if 'study_deck_index' not in st.session_state or not (0 <= st.session_state.study_deck_index < total_cards):
        st.session_state.study_deck_index = 0; st.session_state.study_card_flipped = False
    current_index = st.session_state.study_deck_index
    current_card_id = deck_cards_ids[current_index]['id']
    card_data = run_query("SELECT id, word_phrase, image_filename FROM flashcards WHERE id = ?", (current_card_id,), fetchone=True)
    if not card_data: module_logger.error(f"Card data missing for ID {current_card_id} in deck!"); st.error(f"Card data missing for ID {current_card_id} in deck!"); st.session_state.current_view_mode = 'list_cards'; st.rerun(); return
    
    active_space_name_row = run_query("SELECT name FROM card_spaces WHERE id = ?", (space_id,), fetchone=True)
    active_space_name = active_space_name_row['name'] if active_space_name_row else "Deck"
    
    st.subheader(f"📚 Studying Deck: {html.escape(active_space_name)} (Card {current_index + 1} / {total_cards})")
    st.markdown(f"### Current Card: **{html.escape(card_data['word_phrase'])}**"); st.markdown("---")
    if 'study_card_flipped' not in st.session_state: st.session_state.study_card_flipped = False

    col_card_area, col_image_area = st.columns([0.65, 0.35])
    with col_card_area:
        card_container_height = "450px" 
        if not st.session_state.study_card_flipped:
            front_html_style = f"""border: 2px solid #718096; background-color: #FFFFFF; color: #2D3748; padding: 20px; text-align: center; height: {card_container_height}; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 10px; font-size: 2.8em; font-weight: bold; box-shadow: 0 6px 12px rgba(0,0,0,0.15);"""
            st.markdown(f"<div style='{front_html_style}'>{html.escape(card_data['word_phrase'])}</div>", unsafe_allow_html=True)
        else:
            back_html_style = f"""border: 2px solid #38B2AC; background-color: #F7FAFC; color: #2D3748; padding: 20px; height: {card_container_height}; border-radius: 10px; overflow-y: auto; box-shadow: 0 6px 12px rgba(0,0,0,0.15);"""
            back_html_content = f"<h4 style='margin-top:0; color: #2C5282; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px;'>{html.escape(card_data['word_phrase'])} - Details:</h4>"
            card_fields = run_query("SELECT field_type, content FROM flashcard_fields WHERE flashcard_id = ? ORDER BY field_type, sort_order ASC", (current_card_id,), fetchall=True)
            field_content_map = {}
            for field_row in card_fields:
                if field_row['field_type'] not in field_content_map: field_content_map[field_row['field_type']] = []
                field_content_map[field_row['field_type']].append(field_row['content'])
            for field_key_local, config_local in field_types_config.items():
                contents = field_content_map.get(field_key_local, [])
                if contents:
                    back_html_content += f"<div style='margin-top: 15px;'><strong style='color: #4A5568; font-size: 1.1em;'>{html.escape(config_local['label'])}:</strong></div><ul style='margin-top: 5px; margin-bottom:15px; padding-left: 25px; list-style-type: none;'>"
                    for entry_content_str in contents:
                        content_display = entry_content_str; item_style = "background-color: #EDF2F7; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"
                        if field_key_local == "pairwise_dialogue":
                            try: dialogue_data = json.loads(content_display); q = html.escape(dialogue_data.get('question', 'N/A')); a = html.escape(dialogue_data.get('answer', 'N/A')); content_display = f"<div style='{item_style}'><strong style='color:#3182CE;'>Q:</strong> {q}<br><strong style='color:#3182CE;'>A:</strong> {a}</div>"
                            except Exception as e: content_display = f"<div style='{item_style} background-color: #FFF5F5; color: #C53030;'><i>Invalid dialogue:</i> <pre>{html.escape(content_display)}</pre></div>"; module_logger.error(f"DeckView: Error displaying pairwise_dialogue: {e}")
                        elif field_key_local == "word_family":
                            try:
                                parsed_json = json.loads(content_display)
                                wf_html = ""
                                for key, value_list in parsed_json.items():
                                    disp_key = html.escape(key.replace('_', ' ')).capitalize()
                                    if isinstance(value_list, list) and value_list and any(str(v).strip() for v in value_list):
                                        wf_html += f"<div style='margin-bottom:3px;'><strong style='color:#007A5E;'>{disp_key}:</strong> {', '.join(html.escape(str(v)) for v in value_list if str(v).strip())}</div>"
                                    elif isinstance(value_list, str) and value_list.strip():
                                        wf_html += f"<div style='margin-bottom:3px;'><strong style='color:#007A5E;'>{disp_key}:</strong> {html.escape(value_list)}</div>"
                                no_terms_span = "<span style='font-style:italic; color: #718096;'>No terms provided.</span>"
                                inner_html_for_wf = wf_html if wf_html else no_terms_span
                                content_display = f"<div style='{item_style} background-color: #E6FFFA;'>{inner_html_for_wf}</div>"
                            except Exception as e_wf:
                                content_display = f"<div style='{item_style} background-color: #FFF5F5; color: #C53030;'><i>Invalid Word Family JSON:</i> <pre>{html.escape(content_display)}</pre></div>"
                                module_logger.error(f"DeckView: Error displaying word_family: {e_wf}")
                        elif config_local['json_output']: 
                             try: parsed_json = json.loads(content_display); content_display = f"<pre style='{item_style} white-space: pre-wrap; word-wrap: break-word; background-color: #F0F0F0;'>{html.escape(json.dumps(parsed_json, indent=2))}</pre>"
                             except Exception as e: content_display = f"<pre style='{item_style} white-space: pre-wrap; word-wrap: break-word; background-color: #FFF5F5; color: #C53030;'><i>Invalid JSON:</i> {html.escape(content_display)}</pre>"; module_logger.error(f"DeckView: Error displaying other JSON: {e}")
                        else: content_display = f"<div style='{item_style}'>{html.escape(content_display).replace(chr(10), '<br>')}</div>"
                        back_html_content += f"<li>{content_display}</li>"
                    back_html_content += "</ul>"
            st.markdown(f"<div style='{back_html_style}'>{back_html_content}</div>", unsafe_allow_html=True)
    with col_image_area:
        if card_data['image_filename']:
            img_full_path = os.path.join(MEDIA_DIR, space_id, card_data['image_filename'])
            if os.path.exists(img_full_path): st.image(img_full_path, use_container_width=True, caption="Attached Image")
            else: module_logger.warning(f"Deck Image file not found: {img_full_path}"); st.markdown(f"<div style='height: {card_container_height}; display:flex; align-items:center; justify-content:center; border: 2px dashed #CBD5E0; border-radius:10px; color: #A0AEC0; text-align:center;'>Image not found<br><small>{html.escape(card_data['image_filename'])}</small></div>", unsafe_allow_html=True)
        else: st.markdown(f"<div style='height: {card_container_height}; display:flex; align-items:center; justify-content:center; border: 2px dashed #CBD5E0; border-radius:10px; color: #A0AEC0;'>No Image Attached</div>", unsafe_allow_html=True)

    st.markdown("---")
    nav_cols = st.columns([1, 1, 1])
    with nav_cols[0]:
        if st.button("⬅️ Previous", disabled=(current_index <= 0), key=f"prev_deck_btn_{current_card_id}", use_container_width=True):
            st.session_state.study_deck_index -= 1; st.session_state.study_card_flipped = False; st.rerun()
    with nav_cols[1]:
        if st.button("Flip Card 🃏", key=f"flip_deck_btn_{current_card_id}", use_container_width=True):
            st.session_state.study_card_flipped = not st.session_state.study_card_flipped; st.rerun()
    with nav_cols[2]:
        if st.button("Next ➡️", disabled=(current_index >= total_cards - 1), key=f"next_deck_btn_{current_card_id}", use_container_width=True):
            st.session_state.study_deck_index += 1; st.session_state.study_card_flipped = False; st.rerun()
    if st.button("Back to Card List", key=f"deck_study_back_to_list_btn_{space_id}", use_container_width=True):
        st.session_state.current_view_mode = 'list_cards'; st.session_state.studying_deck_space_id = None; st.session_state.study_deck_index = 0; st.session_state.study_card_flipped = False; st.rerun()
    st.markdown("---")
    
    st.subheader(f"🎓 Live Tutor for: {html.escape(card_data['word_phrase'])}")
    tutor_state_key_base = f"tutor_state_{current_card_id}" 
    render_live_tutor_tabs(card_data, current_card_id, tutor_state_key_base, generate_with_llm_func_ref)

# --- render_list_flashcards_view ---
def render_list_flashcards_view(space_id: str):
    module_logger.debug(f"Rendering LIST CARDS view for space_id: {space_id}") # Corrected
    # ... (Full function as provided in "Aiming for final version")
    # Ensure its internal logger calls also use module_logger
    with st.expander("➕ Add New Flashcard Stub", expanded=True):
        with st.form("new_flashcard_stub_form_v2", clear_on_submit=True): # Unique form key
            word_phrase_input = st.text_input("Word or Phrase for New Card*", key="fc_stub_word_list_view_v2")
            submitted_stub = st.form_submit_button("Create Flashcard Stub & Edit")
            if submitted_stub and word_phrase_input.strip():
                existing_card = run_query("SELECT id FROM flashcards WHERE card_space_id = ? AND lower(word_phrase) = ?", (space_id, word_phrase_input.strip().lower()), fetchone=True)
                if existing_card: st.warning(f"Flashcard '{word_phrase_input.strip()}' already exists.")
                else: flashcard_id = str(uuid.uuid4()); now = int(time.time()); run_query("INSERT INTO flashcards (id, card_space_id, word_phrase, created_at, last_updated_at) VALUES (?, ?, ?, ?, ?)", (flashcard_id, space_id, word_phrase_input.strip(), now, now), commit=True); module_logger.info(f"Flashcard stub created: {flashcard_id}"); st.session_state.editing_flashcard_id = flashcard_id; st.session_state.current_view_mode = 'edit_card'; st.rerun()
            elif submitted_stub: st.warning("Word/Phrase cannot be empty.")
    st.markdown("---")
    if st.button("📚 Study All Cards in this Space", key=f"study_all_btn_list_view_v2_{space_id}"): # Unique key
        flashcards_check = run_query("SELECT id FROM flashcards WHERE card_space_id = ?", (space_id,), fetchall=True)
        if flashcards_check: 
            st.session_state.studying_deck_space_id = space_id; st.session_state.study_deck_index = 0; st.session_state.study_card_flipped = False
            st.session_state.current_view_mode = 'study_deck'; st.session_state.studying_flashcard_id = None 
            st.rerun()
        else: st.warning("No cards to study.")

    flashcards_in_space = run_query("SELECT id, word_phrase FROM flashcards WHERE card_space_id = ? ORDER BY created_at DESC", (space_id,), fetchall=True)
    if not flashcards_in_space: st.info("No flashcards in this space yet."); return
    st.subheader("Your Flashcards:")
    for card_data_row in flashcards_in_space:
        card_id = card_data_row['id']; word_phrase = card_data_row['word_phrase']
        col1, col2, col3, col4 = st.columns([0.5, 0.17, 0.17, 0.16])
        with col1: st.markdown(f"**{html.escape(word_phrase)}**")
        with col2:
            if st.button("✏️ Edit", key=f"edit_btn_list_item_render_v2_{card_id}"): # Unique key
                st.session_state.editing_flashcard_id = card_id; st.session_state.current_view_mode = 'edit_card'
                st.session_state.studying_flashcard_id = None; st.session_state.studying_deck_space_id = None; st.rerun()
        with col3:
            if st.button("📖 Study", key=f"study_single_btn_list_item_render_v2_{card_id}"): # Unique key
                st.session_state.studying_flashcard_id = card_id; st.session_state.current_view_mode = 'study_card'
                st.session_state.study_card_flipped = False; st.session_state.studying_deck_space_id = None; st.rerun()
        with col4: 
            confirm_key = f"confirm_delete_card_state_list_render_v2_{card_id}" # Unique key
            if confirm_key not in st.session_state: st.session_state[confirm_key] = False
            if st.session_state[confirm_key]:
                if st.button("✅ Confirm Delete", key=f"confirm_del_btn_list_item_render_v2_{card_id}", type="primary"): run_query("DELETE FROM flashcards WHERE id = ?", (card_id,), commit=True); st.success(f"Card '{word_phrase}' deleted."); st.session_state[confirm_key] = False; st.rerun() # Unique key
                if st.button("❌ Cancel", key=f"cancel_del_btn_list_item_render_v2_{card_id}"): st.session_state[confirm_key] = False; st.rerun() # Unique key
            else:
                if st.button("🗑️ Del", key=f"del_card_btn_list_item_render_v2_{card_id}", help="Delete card"):  st.session_state[confirm_key] = True; st.rerun() # Unique key
        st.divider()

# --- render_edit_flashcard_view ---
def render_edit_flashcard_view(card_id: str, space_id: str, generate_with_llm_func_ref):
    module_logger.debug(f"Rendering EDIT CARD view for card_id: {card_id}") # Corrected
    # ... (Full function as provided in "Aiming for final version")
    # Ensure its internal logger calls also use module_logger
    # and LLM calls use generate_with_llm_func_ref
    card_data = run_query("SELECT id, word_phrase, image_filename FROM flashcards WHERE id = ? AND card_space_id = ?", (card_id, space_id), fetchone=True)
    if not card_data: module_logger.error(f"Flashcard not found for edit: {card_id}"); st.error("Flashcard not found."); st.session_state.current_view_mode = 'list_cards'; st.session_state.editing_flashcard_id = None; st.rerun(); return
    st.subheader(f"✏️ Editing Card: {html.escape(card_data['word_phrase'])}")
    current_image_filename = card_data['image_filename']; image_bytes_for_llm_card_level = None 
    if current_image_filename:
        img_full_path = os.path.join(MEDIA_DIR, space_id, current_image_filename); 
        if os.path.exists(img_full_path): st.image(img_full_path, caption=current_image_filename, width=150); 
        try: 
            with open(img_full_path, "rb") as f_img: image_bytes_for_llm_card_level = f_img.read()
        except Exception as e_read: module_logger.error(f"Error reading image {img_full_path} for LLM: {e_read}")
        if st.button("Remove Image", key=f"remove_img_btn_edit_view_render_v2_{card_id}"):  # Unique key
            try: 
                if os.path.exists(img_full_path): os.remove(img_full_path)
                run_query("UPDATE flashcards SET image_filename = NULL, last_updated_at = ? WHERE id = ?", (int(time.time()), card_id), commit=True); st.rerun()
            except OSError as e_os: st.error(f"Error removing image file: {e_os}")
    uploaded_image_file_edit = st.file_uploader("Change/Upload Image", type=["jpg", "jpeg", "png"], key=f"edit_img_uploader_widget_view_render_v2_{card_id}") # Unique key
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
                    display_content_loop = content_entry_loop['content']; llm_used_info = content_entry_loop['llm_model_used'] or 'Manual'
                    if config_loop['json_output']:
                        try: parsed_json = json.loads(display_content_loop); st.json(parsed_json); st.caption(f"*(LLM: {llm_used_info})*")
                        except json.JSONDecodeError: st.code(display_content_loop, language='text'); st.caption(f"*(LLM: {llm_used_info} - Invalid JSON)*")
                    else: st.markdown(f"- {html.escape(display_content_loop)} *(LLM: {llm_used_info})*")
                with col2:
                    if st.button("🗑️", key=f"del_edit_field_btn_view_item_render_v2_{field_key_loop}_{content_entry_loop['id']}", help="Delete entry"): run_query("DELETE FROM flashcard_fields WHERE id = ?", (content_entry_loop['id'],), commit=True); st.rerun() # Unique key
            with st.form(key=f"form_user_edit_add_view_item_render_v2_{field_key_loop}_{card_id}", clear_on_submit=True): # Unique key
                default_user_input = config_loop.get('template_for_user', ""); text_area_height = 150 if config_loop.get('json_output', False) else 75
                user_field_input_loop = st.text_area(label=f"Your {config_loop['label'].split('(')[0].strip()}", value=default_user_input, height=text_area_height, key=f"user_edit_input_ta_view_item_render_v2_{field_key_loop}_{card_id}") # Unique key
                if st.form_submit_button(f"Add My Entry"):
                    if user_field_input_loop.strip():
                        if config_loop.get('json_output', False):
                            try: json.loads(user_field_input_loop.strip()) 
                            except json.JSONDecodeError: st.error("Invalid JSON format."); st.stop() 
                        field_id_loop = str(uuid.uuid4()); run_query("INSERT INTO flashcard_fields (id, flashcard_id, field_type, content, llm_model_used, sort_order) VALUES (?, ?, ?, ?, ?, ?)", (field_id_loop, card_id, field_key_loop, user_field_input_loop.strip(), "user_manual", len(existing_contents_loop)), commit=True); st.success(f"Your entry added!"); st.rerun()
                    else: st.warning("Input cannot be empty.")
            show_llm_button = config_loop['allow_multiple'] or not any(ec['llm_model_used'] != "user_manual" and ec['llm_model_used'] is not None and ec['field_type'] == field_key_loop for ec in existing_contents_loop)
            if show_llm_button:
                if st.button(f"Generate with LLM", key=f"gen_llm_edit_btn_view_item_render_v2_{field_key_loop}_{card_id}"): # Unique key
                    with st.spinner(f"Generating {config_loop['label']}..."):
                        user_prompt_for_llm_loop = config_loop['prompt_template'].replace("{WORD_PHRASE}", card_data['word_phrase'])
                        current_image_for_llm_field_loop = image_bytes_for_llm_card_level if field_key_loop in ["example_sentence", "definition", "pairwise_dialogue"] else None
                        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                        generated_content_loop = loop.run_until_complete(generate_with_llm_func_ref(system_prompt="You are an expert English language learning assistant.", user_prompt=user_prompt_for_llm_loop, image_bytes=current_image_for_llm_field_loop, output_json=config_loop['json_output']))
                        try: loop.close()
                        except Exception as e_loop_close: module_logger.warning(f"Error closing event loop in edit view: {e_loop_close}")
                        
                        if generated_content_loop and isinstance(generated_content_loop, dict) and "error" in generated_content_loop:
                            st.error(f"LLM Error: {generated_content_loop.get('error')}. Raw: {generated_content_loop.get('raw_output','')[:200]}")
                        elif generated_content_loop:
                            if config_loop['json_output'] and not isinstance(generated_content_loop, dict): st.error(f"LLM did not return valid JSON. Received: {str(generated_content_loop)[:200]}")
                            else: 
                                content_to_save_loop = json.dumps(generated_content_loop, indent=4) if config_loop['json_output'] and isinstance(generated_content_loop, dict) else str(generated_content_loop)
                                field_id_llm_loop = str(uuid.uuid4()); model_used_loop = "Gemini" if st.session_state.use_gemini and st.session_state.get('gemini_llm_instance') else st.session_state.ollama_model
                                run_query("INSERT INTO flashcard_fields (id, flashcard_id, field_type, content, llm_model_used, sort_order) VALUES (?, ?, ?, ?, ?, ?)", (field_id_llm_loop, card_id, field_key_loop, content_to_save_loop, model_used_loop, len(existing_contents_loop)), commit=True); st.success(f"LLM generated entry added!"); st.rerun()
                        else: st.error(f"LLM failed to generate (no content returned or unexpected response).")
    if st.button("Done Editing (Back to List)", key=f"done_edit_back_to_list_btn_view_render_v2_{card_id}"):  # Unique key
        st.session_state.current_view_mode = 'list_cards'; st.session_state.editing_flashcard_id = None; st.rerun()
