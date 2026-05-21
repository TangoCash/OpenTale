# OpenTale - AI Book Writer

A web-based application that guides you through the process of writing a book with AI assistance. The system uses AI models (OpenAI-compatible APIs) to help generate world settings, characters, outlines, and complete chapters.

## Features

- Web-based user interface with no authentication required
- Step-by-step guided book writing process
- Real-time AI generation of:
  - World settings and environments
  - Character profiles and development
  - Book outlines with chapter structure
  - Scene generation for individual chapters
  - Full chapter content
- Editable configuration via web UI (no manual `.env` file needed)
- Model selector — fetch available models directly from the API
- Connection tester — verify your API endpoint is reachable
- Progress tracking
- Ability to edit and save generated content
- All content stored in local files for easy access
- Multi-project support — work on multiple books simultaneously

## Architecture

The application consists of:

- **Flask Web Server**: Provides the user interface and manages the book generation process
- **AI Agents**: Specialized agents for different aspects of book creation:
  - Story planning
  - World building
  - Character development
  - Scene creation
  - Writing and editing
- **Prompt Management**: Centralized prompt templates in `prompts.py`
- **File Storage**: Local storage of all generated content in the `book_output` directory

## Quick Start

### Windows

Double-click `start.bat` — it will:
1. Create a Python virtual environment (`.venv`) if it doesn't exist
2. Install all dependencies
3. Launch the web application

Or run from the command line:
```cmd
start.bat
```

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

The `start.sh` script does the same as `start.bat` — creates `.venv`, installs dependencies, and runs the app.

### Manual Setup

If you prefer to do things manually:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python web_app.py
```

## Usage

1. Start the web application (see **Quick Start** above).
2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```
3. Go to the **Configuration** page (`/config`) to set up your AI provider:
   - Enter the **Base URL** (e.g. `https://openrouter.ai/api/v1` or `http://localhost:11434/v1` for Ollama)
   - Enter your **API Key** (if required)
   - Click **Test** to verify the connection — the model list will auto-populate on success
   - Select your preferred **Model**
   - Adjust temperature, max tokens, etc.
   - Click **Save Configuration**

4. Follow the step-by-step writing workflow:
   - Create a synopsis
   - Build a world setting
   - Generate characters
   - Create a book outline
   - Work chapter by chapter to generate your book

## Book Writing Workflow

The application guides you through a logical book creation process:

1. **Synopsis**: Define the core idea and story direction
2. **World Building**: Define the setting, time period, and environment for your story
3. **Character Creation**: Generate the main characters for your book
4. **Outline Generation**: Create a chapter-by-chapter outline of your story
5. **Chapter Writing**:
   - Generate action beats for a chapter
   - Generate a complete chapter
   - Edit and refine using the editor agent
   - Proceed to the next chapter

## Output Structure

All generated content is saved in the `book_output` directory:
```
book_output/
├── .active_project           # Currently active project name
├── (project_name)/
│   ├── world.txt             # World setting
│   ├── characters.txt        # Character profiles
│   ├── synopsis.txt          # Story synopsis
│   ├── outline.txt           # Full book outline
│   ├── outline.json          # Structured outline data
│   ├── chapters.json         # Chapter metadata
│   ├── master_prompt.txt     # Master writing prompt
│   ├── settings.json         # Project-specific settings
│   └── chapters/
│       ├── chapter_1.txt
│       ├── chapter_1_editor.txt       # Editor-reviewed version
│       ├── chapter_1_action_beats.txt # Action beats for the chapter
│       ├── chapter_2.txt
│       └── ...
```

## Configuration

All AI provider settings are stored in `config.json` in the project root. This file is auto-created with defaults on first run and can be edited at any time via the web UI (`/config` page).

### Configurable Parameters

| Parameter     | Description                                        | Default                          |
|---------------|----------------------------------------------------|----------------------------------|
| `base_url`    | API endpoint URL                                   | `https://openrouter.ai/api/v1`   |
| `api_key`     | API key (stored locally in config.json)            | (empty)                          |
| `model`       | Model identifier                                   | `google/gemini-2.5-flash`        |
| `temperature` | Controls randomness (0.0 - 2.0)                    | `0.7`                            |
| `top_p`       | Nucleus sampling threshold (0.0 - 1.0)             | `1.0`                            |
| `max_tokens`  | Maximum tokens per response                        | `10000`                          |
| `seed`        | Random seed for reproducible outputs               | `42`                             |
| `timeout`     | API request timeout in seconds                     | `1000`                           |
| `debug`       | Enable prompt debugging (saves prompts to disk)    | `false`                          |

### Changing Projects

You can create and switch between multiple book projects from the web interface using the project selector in the navigation bar.

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- An OpenAI-compatible API endpoint (e.g. OpenRouter, Ollama, LM Studio, etc.)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.