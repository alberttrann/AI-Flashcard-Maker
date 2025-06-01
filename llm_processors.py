# llm_processors.py
import json
import httpx # For Ollama
from google import genai
from google.genai import types as google_genai_types
import logging
import asyncio
import inspect # For debugging module loading

logger = logging.getLogger(__name__)
logger.info(f"llm_processors.py loaded. GeminiAPIProcessor will be defined in this module. Source: {__file__}")

# --- OLLAMA LLAMA PROCESSOR ---
class OllamaLlamaProcessor:
    def __init__(self, base_url="http://localhost:11434", model_name="llama3.1:8b"):
        self.base_url = base_url.strip() if base_url else "http://localhost:11434"
        self.model_name = model_name.strip() if model_name else "llama3.1:8b"
        self.api_url = f"{self.base_url}/api/generate"
        logger.info(f"OllamaLlamaProcessor Inst:{id(self)} initialized for model: {self.model_name} at {self.api_url}")
        # Log signature for clarity
        if hasattr(self, 'generate_text'):
            sig = inspect.signature(self.generate_text)
            logger.debug(f"OllamaLlamaProcessor Inst:{id(self)} generate_text signature: {sig}")


    async def generate_text(self, system_prompt: str, user_prompt: str, output_format_json: bool = False) -> str | dict | None:
        payload = {
            "model": self.model_name,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
        }
        if output_format_json:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                
                generated_content = response_data.get("response", "")
                if output_format_json:
                    try:
                        return json.loads(generated_content)
                    except json.JSONDecodeError:
                        logger.error(f"OllamaLlamaProcessor: Failed to parse JSON from: {generated_content}")
                        return {"error": "Failed to parse LLM JSON output", "raw_output": generated_content}
                return generated_content.strip()
        except httpx.RequestError as e:
            logger.error(f"OllamaLlamaProcessor: Request error to Ollama API: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"OllamaLlamaProcessor: HTTP status error from Ollama API: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"OllamaLlamaProcessor: Unexpected error: {e}", exc_info=True)
        return None

# --- GEMINI API PROCESSOR ---
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
            self.client = None 
            raise 
        
        self.default_model_name_str = "gemini-2.0-flash"
        self.generation_count = 0
        logger.info(f"GeminiAPIProc Inst:{self.instance_id_log} initialized. Default model string: {self.default_model_name_str}")
        # Log signature for clarity
        if hasattr(self, 'generate_text'):
            sig = inspect.signature(self.generate_text)
            logger.debug(f"GeminiAPIProc Inst:{self.instance_id_log} generate_text signature: {sig}")


    async def _generate_content_with_gemini_using_client_models_stream(
        self, 
        model_name_str_to_use: str, 
        current_contents_for_api: list, 
        generation_config_obj: google_genai_types.GenerateContentConfig,
    ):
        loop = asyncio.get_event_loop()
        
        def sync_call_gemini_stream_and_collect():
            full_text_response = ""
            if not self.client:
                logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Client not available in sync_call_gemini_stream_and_collect")
                return "[Error: Gemini client not available]"

            response_stream = self.client.models.generate_content_stream(
                model=model_name_str_to_use,
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
        try:
            return await loop.run_in_executor(None, sync_call_gemini_stream_and_collect)
        except Exception as e:
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Error in run_in_executor for Gemini stream: {e}", exc_info=True)
            return f"[Error: Gemini stream processing failed - {type(e).__name__}]"


    async def generate_text(self, system_prompt: str, user_prompt: str, image_bytes: bytes = None, output_format_json: bool = False, rag_context: str = ""):
        self.generation_count += 1
        model_str_to_use = self.default_model_name_str
        log_prefix = f"GeminiAPIProc Inst:{self.instance_id_log} GenCall {self.generation_count} (ModelStr: {model_str_to_use}):"

        if not self.client:
            logger.error(f"{log_prefix} Gemini client not initialized.")
            return "[Error: Gemini client not initialized]"

        try:
            parts_list = []
            full_user_text_prompt = user_prompt
            if rag_context:
                full_user_text_prompt = f"Relevant Context:\n{rag_context}\n\n---\n\nUser Query:\n{user_prompt}"
            
            parts_list.append(google_genai_types.Part(text=full_user_text_prompt))
            if image_bytes:
                try:
                    image_part_data = {"mime_type": "image/jpeg", "data": image_bytes}
                    parts_list.append(google_genai_types.Part(inline_data=image_part_data))
                except Exception as e_img:
                    logger.error(f"{log_prefix} Error creating image part for Gemini chat: {e_img}")
            
            current_contents_for_api = [google_genai_types.Content(role="user", parts=parts_list)]

            response_mime = "application/json" if output_format_json else "text/plain"
            gen_config_dict = {"response_mime_type": response_mime}
            
            effective_system_prompt = system_prompt 
            if effective_system_prompt:
                gen_config_dict["system_instruction"] = [google_genai_types.Part.from_text(text=effective_system_prompt)]
            
            tools = [google_genai_types.Tool(google_search=google_genai_types.GoogleSearch())] if not output_format_json else None
            if tools:
                gen_config_dict["tools"] = tools

            generation_config_obj = google_genai_types.GenerateContentConfig(**gen_config_dict)

            logger.debug(f"{log_prefix} Calling _generate_content_with_gemini_using_client_models_stream. System: '{effective_system_prompt[:50]}...', User: '{user_prompt[:50]}...', Image: {image_bytes is not None}, OutputJSON: {output_format_json}, RAG: {bool(rag_context)}")

            generated_text = await self._generate_content_with_gemini_using_client_models_stream(
                model_name_str_to_use=model_str_to_use,
                current_contents_for_api=current_contents_for_api,
                generation_config_obj=generation_config_obj
            )

            if output_format_json and generated_text and not generated_text.startswith("[Error:"):
                try: 
                    return json.loads(generated_text)
                except json.JSONDecodeError: 
                    logger.error(f"{log_prefix} Failed to parse JSON from: {generated_text}")
                    return {"error": "LLM JSON parse error", "raw_output": generated_text}
            
            return generated_text

        except Exception as e:
            logger.error(f"{log_prefix} Error in generate_text setup: {e}", exc_info=True)
            return f"[Error: Gemini processing setup failed - {type(e).__name__}]"