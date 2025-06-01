# llm_processors.py
import json
import httpx
from google import genai
from google.genai import types as google_genai_types
import logging
import asyncio
# import inspect # Not strictly needed for runtime, can be for debugging

# Use Streamlit's logger for consistency if running within Streamlit context
# If this file might be used outside Streamlit, standard logging is fine
try:
    import streamlit as st
    logger = st.logger.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class OllamaLlamaProcessor:
    def __init__(self, base_url="http://localhost:11434", model_name="llama3.1:8b"):
        self.base_url = base_url.strip() if base_url else "http://localhost:11434"
        self.model_name = model_name.strip() if model_name else "llama3.1:8b"
        self.api_url = f"{self.base_url}/api/generate"
        logger.debug(f"OllamaLlamaProcessor Inst:{id(self)} initialized for model: {self.model_name} at {self.api_url}")

    async def generate_text(self, system_prompt: str, user_prompt: str, output_format_json: bool = False) -> str | dict | None:
        payload = {"model": self.model_name, "prompt": user_prompt, "system": system_prompt, "stream": False}
        if output_format_json: 
            payload["format"] = "json"
        
        logger.debug(f"Ollama Request Payload: {payload}")
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(self.api_url, json=payload)
                logger.debug(f"Ollama Response Status: {response.status_code}")
                response.raise_for_status()
                response_data = response.json()
                logger.debug(f"Ollama Response Data: {response_data}")
                
                generated_content = response_data.get("response", "")
                if not generated_content and "error" in response_data: # Ollama might return error in response field
                     logger.error(f"Ollama API returned an error in response: {response_data['error']}")
                     return {"error": f"Ollama API error: {response_data['error']}", "raw_output": json.dumps(response_data)}

                if output_format_json:
                    try:
                        parsed_json = json.loads(generated_content)
                        return parsed_json
                    except json.JSONDecodeError as e:
                        logger.error(f"OllamaLlamaProcessor: Failed to parse JSON from: {generated_content}. Error: {e}")
                        return {"error": "Failed to parse LLM JSON output", "raw_output": generated_content}
                return generated_content.strip()
        except httpx.RequestError as e:
            logger.error(f"OllamaLlamaProcessor: Request error to Ollama API: {e}", exc_info=True)
            return {"error": f"Ollama request error: {e}", "raw_output": ""}
        except httpx.HTTPStatusError as e:
            logger.error(f"OllamaLlamaProcessor: HTTP status error from Ollama API: {e.response.status_code} - {e.response.text}", exc_info=True)
            return {"error": f"Ollama HTTP error {e.response.status_code}: {e.response.text}", "raw_output": e.response.text}
        except Exception as e:
            logger.error(f"OllamaLlamaProcessor: Unexpected error: {e}", exc_info=True)
            return {"error": f"Ollama unexpected error: {e}", "raw_output": ""}


class GeminiAPIProcessor:
    def __init__(self, api_key: str):
        self.instance_id_log = str(id(self))[-6:]
        if not api_key: 
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Initialization failed: API key is missing.")
            raise ValueError("Gemini API key is required for GeminiAPIProcessor.")
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(f"GeminiAPIProc Inst:{self.instance_id_log} GenAI client CREATED with API key.")
        except Exception as e: 
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Failed to create GenAI client: {e}", exc_info=True)
            self.client = None; raise
        self.default_model_name_str = "gemini-2.0-flash" 
        self.generation_count = 0
        logger.debug(f"GeminiAPIProc Inst:{self.instance_id_log} initialized. Default model: {self.default_model_name_str}")

    async def _generate_content_with_gemini_using_client_models_stream(self, model_name_str_to_use: str, current_contents_for_api: list, generation_config_obj: google_genai_types.GenerateContentConfig):
        loop = asyncio.get_event_loop()
        def sync_call_gemini_stream_and_collect():
            full_text_response = ""
            if not self.client: 
                logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Client not available in sync_call")
                return "[Error: Gemini client not available]"
            try:
                response_stream = self.client.models.generate_content_stream(
                    model=model_name_str_to_use, 
                    contents=current_contents_for_api, 
                    config=generation_config_obj # This contains response_mime_type="text/plain"
                )
                for chunk in response_stream: # The error happens here, before we can access chunk.text
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
            except json.JSONDecodeError as e_json_stream: # Explicitly catch JSONDecodeError here
                logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} JSONDecodeError during content streaming: {e_json_stream}. This can happen if text/plain is expected but chunks are parsed as JSON by the lib.", exc_info=True)
                # If we expected text, we might have accumulated something before the error.
                # Or, the entire response was just text and the lib failed on the first non-JSON chunk.
                # This is tricky because the error happens *inside* the library's iteration.
                # For now, if this specific error happens, we assume the accumulated text (if any) might be it,
                # or we return an error indicating a streaming/parsing problem.
                if full_text_response: # If some text was collected before the JSON error
                    logger.warning(f"GeminiAPIProc Inst:{self.instance_id_log} Returning partially collected text due to JSONDecodeError in stream: '{full_text_response[:100]}...'")
                    return full_text_response.strip() 
                return f"[Error: Gemini stream parsing error (JSONDecodeError) - likely for text/plain response]"

            except Exception as e_stream: 
                logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Error during content streaming: {e_stream}", exc_info=True)
                return f"[Error: Gemini content streaming failed - {type(e_stream).__name__}]"

        try: 
            return await loop.run_in_executor(None, sync_call_gemini_stream_and_collect)
        except Exception as e_exec: 
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Error in run_in_executor for Gemini stream: {e_exec}", exc_info=True)
            return f"[Error: Gemini stream executor failed - {type(e_exec).__name__}]"


    async def generate_text(self, system_prompt: str, user_prompt: str, image_bytes: bytes = None, output_format_json: bool = False, rag_context: str = ""):
        self.generation_count += 1
        model_str_to_use = self.default_model_name_str
        log_prefix = f"GeminiAPIProc Inst:{self.instance_id_log} GenCall {self.generation_count} (Model: {model_str_to_use}):"

        if not self.client: 
            logger.error(f"{log_prefix} Gemini client not initialized.")
            return {"error": "Gemini client not initialized.", "raw_output": ""}
        
        try:
            full_user_text_prompt = user_prompt
            if rag_context and rag_context.strip():
                full_user_text_prompt = f"Relevant Context:\n{rag_context}\n\n---\n\nUser Query:\n{user_prompt}"
            
            parts_list = [google_genai_types.Part(text=full_user_text_prompt)]
            if image_bytes: 
                parts_list.append(google_genai_types.Part(inline_data={"mime_type": "image/jpeg", "data": image_bytes}))
            
            current_contents_for_api = [google_genai_types.Content(role="user", parts=parts_list)]
            
            gen_config_dict = {"response_mime_type": "application/json" if output_format_json else "text/plain"}
            if system_prompt: 
                gen_config_dict["system_instruction"] = [google_genai_types.Part.from_text(text=system_prompt)]
            
            # Enable search only for non-JSON, text-based generation to avoid interference
            if not output_format_json : 
                gen_config_dict["tools"] = [google_genai_types.Tool(google_search=google_genai_types.GoogleSearch())]
            
            generation_config_obj = google_genai_types.GenerateContentConfig(**gen_config_dict)
            
            logger.debug(f"{log_prefix} Calling Gemini. OutputJSON: {output_format_json}, System: '{system_prompt[:50]}...', User: '{user_prompt[:50]}...'")
            
            generated_text_response = await self._generate_content_with_gemini_using_client_models_stream(
                model_str_to_use, 
                current_contents_for_api, 
                generation_config_obj
            )

            # Check if the streaming itself returned an error string
            if isinstance(generated_text_response, str) and generated_text_response.startswith("[Error:"):
                logger.error(f"{log_prefix} Error from Gemini stream helper: {generated_text_response}")
                return {"error": generated_text_response, "raw_output": ""}

            if output_format_json:
                if not generated_text_response or not generated_text_response.strip().startswith(("{", "[")): # Basic check for JSON
                    logger.error(f"{log_prefix} Expected JSON but got: {generated_text_response}")
                    return {"error": "LLM did not return valid JSON format.", "raw_output": generated_text_response}
                try: 
                    return json.loads(generated_text_response)
                except json.JSONDecodeError as e: 
                    logger.error(f"{log_prefix} Failed to parse JSON from: {generated_text_response}. Error: {e}", exc_info=True)
                    return {"error": "Failed to parse LLM JSON output", "raw_output": generated_text_response}
            
            return generated_text_response # Should be a string here

        except Exception as e: # Catch other unexpected errors during setup
            logger.error(f"{log_prefix} Unexpected error in generate_text setup: {e}", exc_info=True)
            return {"error": f"Gemini processing setup failed: {e}", "raw_output": ""}
