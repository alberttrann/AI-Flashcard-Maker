# Ai Flashcard Maker: AI-Powered English Learning Flashcard Generator

Ai Flashcard Maker is an intelligent flashcard application designed to help English language learners create, manage, and study vocabulary and phrases with the power of Large Language Models (LLMs). It goes beyond traditional flashcard apps by leveraging AI to generate rich content, provide interactive tutoring, and offer deeper insights into language usage.

The application is built with Python, Streamlit for the user interface, and supports both local Ollama models (like Llama 3.1 8B) and Google's Gemini API for its AI capabilities.

## Features

### I. Flashcard Creation & Management
*   **Card Spaces:** Organize flashcards into different "spaces" or decks based on topics, lessons, or difficulty.
*   **Core Flashcard Fields:**
    *   **Word/Phrase:** The primary term to learn.
    *   **Image Attachment:** Associate an image with each flashcard for visual learning.
*   **LLM-Assisted Content Generation:** For each flashcard, the LLM can help generate:
    *   **Definitions:** Clear and concise explanations of the word/phrase.
    *   **Word Family:** Related words (nouns, verbs, adjectives, adverbs) derived from the same root (e.g., normal, normality, normalize).
    *   **Example Sentences:** Contextual examples demonstrating usage.
    *   **Pairwise Dialogues (Q&A):** Short conversations using the word/phrase, formatted as a question and an answer.
*   **User Customization:**
    *   Users can add their own entries for any field.
    *   For fields like Word Family and Pairwise Dialogues, JSON templates are provided to guide users in creating LLM-friendly structured input.
    *   Ability to regenerate LLM content for any field.
    *   Support for multiple definitions and example sentences per card.
*   **Persistent Storage:** All flashcards, card spaces, and associated content (including LLM-generated fields and user entries) are stored persistently in an SQLite database. Images are stored locally on the server.

### II. Study Modes
*   **Single Card Study:** Focus on one card at a time, with a "flip" animation to reveal details. The associated image is always visible.
*   **Deck Study:** Go through all cards in a selected "Card Space" sequentially, with "Previous," "Next," and "Flip" controls.

### III. Live Tutor (LLM-Powered Interactive Learning - per card)
Accessible from both single card and deck study views, focusing on the current flashcard:
1.  **🎯 Tricky Questions:**
    *   LLM generates nuanced multiple-choice questions (MCQs) or open-ended questions designed to test deeper understanding of the word/phrase.
    *   Users can submit answers and receive LLM-generated feedback and explanations.
2.  **📝 MCQs (Standard):**
    *   LLM generates standard multiple-choice questions focusing on basic definitions, synonyms, or common usage.
    *   Users select an answer and get immediate feedback and explanations from the LLM.
3.  **✍️ Write & Grade:**
    *   Users are prompted to write a sentence or short paragraph using the target word/phrase.
    *   The LLM provides a holistic paragraph of feedback covering:
        *   Critique of the target word/phrase usage (context, naturalness).
        *   Key advice on grammar, typos, and coherence.
        *   One or two recommended alternative sentences or improvements.
4.  **🔄 Semantic Synonyms & Antonyms:**
    *   Users can request the LLM to generate semantic synonyms or antonyms for the target word/phrase.
    *   Each generated term is accompanied by a brief explanation of its nuance and an example sentence.
    *   Users can generate multiple synonyms/antonyms one by one.
*   **Persistent Tutor Interactions:** All questions, user answers, LLM feedback, and generated synonyms/antonyms from the Live Tutor sessions are saved to the database, allowing users to review their learning history for each card.

### IV. LLM Configuration
*   **Switchable Models:** Users can choose between:
    *   A local Ollama model (default: Llama 3.1 8B, configurable endpoint and model name).
    *   Google's Gemini API (requires user to provide their API key).
*   Settings are saved locally for persistence across sessions.

## Project Structure

*   **`main_flashcard_app.py`**: Main Streamlit application script, handles UI routing, high-level session state.
*   **`ui_components.py`**: Contains functions for rendering different UI views (list, edit, study single, study deck, Live Tutor tabs).
*   **`tutor_logic.py`**: Handles LLM prompt engineering for tutor features, processing LLM responses for these features, and saving/retrieving tutor interactions from the database.
*   **`llm_processors.py`**: Defines classes (`OllamaLlamaProcessor`, `GeminiAPIProcessor`) to interact with the different LLM backends.
*   **`db_utils.py`**: Manages all SQLite database interactions (initialization, queries, CRUD operations for flashcards and tutor data) and global configurations like `field_types_config`.
*   **`flashcard_db.sqlite`**: The SQLite database file.
*   **`flashcard_media/`**: Directory where uploaded images for flashcards are stored (organized by card space).
*   **`flash_app_settings.json`**: Stores user-specific settings like API keys and Ollama configuration.

## Getting Started

### Prerequisites
*   Python 3.9+
*   Ollama installed and a model (e.g., `llama3.1:8b`) pulled, if using Ollama.
    *   Ensure Ollama server is running (typically `ollama serve`).
*   A Google Gemini API Key, if using the Gemini model.

### Installation
1.  **Clone the repository (if applicable) or set up your project files.**
    ```bash
    # git clone github.com/alberttran/AI-Flashcard-Maker.git
    # cd AI-Flashcard-Maker
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    Create a `requirements.txt` file with the following (you might need to adjust versions or add more based on your exact setup):
    ```
    streamlit
    httpx
    google-generativeai
    Pillow # For image handling, if not implicitly included
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1.  Ensure your Ollama server is running if you plan to use it: `ollama serve`
2.  Run the Streamlit application:
    ```bash
    streamlit run main_flashcard_app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

### Configuration
*   **Ollama:**
    *   The default endpoint is `http://localhost:11434` and model `llama3.1:8b`.
    *   These can be changed in the "Settings" section of the app's sidebar. Changes are saved in `flash_app_settings.json`.
*   **Gemini API:**
    *   Toggle "Use Gemini API" in the sidebar.
    *   Enter your Gemini API Key when prompted. The key is saved (locally, unencrypted in `flash_app_settings.json` for simplicity in this version - **be mindful of security if deploying publicly with pre-filled keys**).

## Future Enhancements / TODO
*   [ ] More sophisticated word selection for Live Tutor in deck view.
*   [ ] Audio generation for example sentences, dialogues, and words/phrases.
*   [ ] Spaced Repetition System (SRS) integration for study scheduling.
*   [ ] User authentication and cloud storage for multi-user/multi-device access.
*   [ ] Advanced search and filtering of flashcards.
*   [ ] More robust error handling and user feedback for LLM generation failures.
*   [ ] UI/UX refinements for even smoother interactions.
*   [ ] Option for user to select different LLM models from a list (if multiple are available on Ollama or via Gemini).
