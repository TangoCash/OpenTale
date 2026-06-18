"""Define the API client for book generation system — powered by LangChain."""

import os
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import prompts

# Constants
PROMPT_DEBUGGING_DIR = "prompt_debugging"


def check_openai_connection(agent_config: Dict):
    """Checks if the OpenAI API connection is valid."""
    try:
        from openai import OpenAI
        config = agent_config["config_list"][0]
        client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
        )
        # Make a cheap call to list models
        client.models.list()
        print("✅ OpenAI API connection successful.")
    except Exception as e:
        print(
            f"❌ OpenAI API connection failed. Please check your API key and configuration. Error: {e}"
        )


class BookAgents:
    def __init__(self, agent_config: Dict, outline: Optional[List[Dict]] = None):
        """Initialize with book outline context."""
        self.agent_config = agent_config
        self.outline = outline
        self.world_elements = {}
        self.character_developments = {}
        self.debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

        config = agent_config["config_list"][0]

        # LangChain ChatOpenAI client — unified interface for all LLM calls
        self.llm = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            temperature=agent_config.get("temperature", 0.7),
            max_tokens=agent_config.get("max_tokens", 10000),
        )

        # Streaming-enabled variant (stream=True forces streaming mode)
        self.streaming_llm = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            temperature=agent_config.get("temperature", 0.7),
            max_tokens=agent_config.get("max_tokens", 10000),
        )

        self.system_prompts: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Helper: format outline context for system prompts
    # ------------------------------------------------------------------
    def _format_outline_context(self) -> str:
        """Format the book outline into a readable context string."""
        if not self.outline:
            return ""
        context_parts = ["Complete Book Outline:"]
        for chapter in self.outline:
            context_parts.extend([
                f"\nChapter {chapter['chapter_number']}: {chapter['title']}",
                chapter["prompt"],
            ])
        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def _save_debug_messages(self, messages: List[Dict], agent_name: str, request_type: str):
        """Saves the request messages for debugging, grouping by role."""
        if not self.debug:
            return
        os.makedirs(PROMPT_DEBUGGING_DIR, exist_ok=True)
        grouped = {}
        for m in messages:
            role = m["role"]
            grouped.setdefault(role, []).append(m["content"])
        for role, contents in grouped.items():
            path = os.path.join(PROMPT_DEBUGGING_DIR, f"{agent_name}_{request_type}_{role}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n---\n\n".join(contents))

    def _wrap_stream_for_debug(self, stream, agent_name: str, response_name: str):
        """Generator wrapper that collects chunks and saves the full response for debugging."""
        if not self.debug:
            yield from stream
            return
        collected = []
        for chunk in stream:
            if chunk.content:
                collected.append(chunk.content)
            yield chunk
        path = os.path.join(PROMPT_DEBUGGING_DIR, f"{agent_name}_{response_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(collected))

    # ------------------------------------------------------------------
    # Build system prompt strings with dynamic outline context
    # ------------------------------------------------------------------
    def _writer_system(self) -> str:
        outline = self._format_outline_context()
        return f"""You are an expert creative writer, a master storyteller who brings scenes to life with breathtaking detail and deep emotional resonance.

Your mission is to write scenes based on the provided outline context and the user's request, 
adhering to the following directives and craft rules at all times.

### Outline Context
{outline}

---
### Core Directives (Non-Negotiable Rules)
1.  **Strict Plot Adherence:** You must follow the provided **Chapter Outline / Action Beats** with absolute precision and in the correct order. Do not add new plot points, deviate from the sequence, or skip any beats. Your task is to bring the provided outline to life.
2.  **Long-Form Chapter:** Chapters are long-form. If you feel you have covered the beats but the chapter is not long enough, expand *within* the existing events (moment-to-moment detail, dialogue, interiority, sensory grounding). Do not rush.
3.  **No Writing Ahead:** You are NOT allowed to "pull in" events from future chapters to make a chapter longer.
4.  **Scene Integrity:** Write a single, complete chapter with a clear beginning, middle, and end as defined by the story beats. Conclude the chapter exactly where the final story beat specifies. Ensure all transitions are smooth and logical.

---
### Craft & Style Rules (Your Authorial Voice)
*   **Show, Don't Tell:** This is your primary storytelling technique. Reveal character, plot, and world-building through character actions, subtext, body language, and sensory information, not exposition.
*   **Prose and Cadence:** Create engaging, dynamic prose. Employ a varied sentence structure, mixing short, punchy sentences for tension with longer, descriptive sentences for atmosphere.
*   **Details Matter:** Use rich, vivid details to immerse the reader. Add a lot of details, and describe the environment and characters where it makes sense.
*   **Authentic, Purposeful Dialogue:** Dialogue must sound like real people talking. Every line must either reveal character, advance the plot, or build tension. Each character's voice must be distinct and consistent with their profile.
*   **Grounded Tone:** Avoid clichés, melodrama, and overly sentimental prose. Keep the emotional expression authentic and grounded.
*   **Forbidden Words:** You are forbidden from using the following words: **peril, fraught, thwart, dire, that, feel/feeling/felt, back, just, then, ail, look, maybe, knew/know**. Use stronger verbs and more descriptive phrasing instead.

---
Always reference the outline and previous content.
Mark drafts with 'SCENE:' and final versions with 'SCENE FINAL:'
"""

    def _editor_system(self) -> str:
        outline = self._format_outline_context()
        return f"""You are an expert editor ensuring quality and consistency.

Your mission is to review and improve the provided chapter content based on the provided outline context and the user's request, 
adhering to the following directives at all times.

### Outline Context
{outline}

---
### Core Directives (Non-Negotiable Rules)
1. Check alignment with outline
2. Verify character consistency
3. Maintain world-building rules
4. Improve prose quality
5. Return complete edited chapter
6. Never ask to start the next chapter, as the next step is finalizing this chapter
7. Each chapter MUST be at least 5000 words.

Format your responses:
1. Start critiques with 'FEEDBACK:'
2. Provide suggestions with 'SUGGEST:'
3. Return full edited chapter with 'EDITED_SCENE:'

---
Always reference specific outline elements in your feedback.
"""

    # ------------------------------------------------------------------
    # Initialize agent system prompts (called externally with context)
    # ------------------------------------------------------------------
    def create_agents(self, initial_prompt: str, num_chapters: int):
        """Set up system prompts for each agent type."""
        outline_context = self._format_outline_context()

        outline_creator_system = f"""Generate a detailed {num_chapters}-chapter outline.

Start with "OUTLINE:" and end with "END OF OUTLINE"

YOU MUST USE EXACTLY THIS FORMAT FOR EACH CHAPTER - NO DEVIATIONS:

Optional: ### [Act 1]: [Act Title]

Chapter 1: [Title]
- Key Events:
    * [Event 1]
    * [Event 2]
    * [Event 3]
    * [Event 4]
    * [Event 5]
- Character Developments: [Specific character moments and changes]
- Setting: [Specific location and atmosphere]
- Tone: [Specific emotional and narrative tone]

Chapter 2: [Title]
- Key Events:
    * [Event 1]
    * [Event 2]
    * [Event 3]
    * [Event 4]
    * [Event 5]
- Character Developments: [Specific character moments and changes]
- Setting: [Specific location and atmosphere]
- Tone: [Specific emotional and narrative tone]

[CONTINUE IN SEQUENCE FOR ALL {num_chapters} CHAPTERS]

CRITICAL REQUIREMENTS:
1. Create EXACTLY {num_chapters} chapters, numbered 1 through {num_chapters} in order
2. NEVER repeat chapter numbers or restart the numbering
3. EVERY chapter must have AT LEAST 5 specific Key Events
4. Maintain a coherent story flow from Chapter 1 to Chapter {num_chapters}
5. Use proper indentation with bullet points for Key Events
6. NO EXCEPTIONS to this format - follow it precisely for all chapters

Initial Premise:
{initial_prompt}
"""

        world_builder_system = f"""You are an expert in world-building who creates rich, consistent settings.
            
Your role is to establish ALL settings and locations needed for the entire story based on a provided story arc.

### Outline Context
{outline_context}

Your responsibilities:
1. Review the story arc to identify every location and setting needed
2. Create detailed descriptions for each setting, including:
- Physical layout and appearance
- Atmosphere and environmental details
- Important objects or features
- Sensory details (sights, sounds, smells)
3. Identify recurring locations that appear multiple times
4. Note how settings might change over time
5. Create a cohesive world that supports the story's themes

Format your response as:
WORLD_ELEMENTS:

[LOCATION NAME]:
- Physical Description: [detailed description]
- Atmosphere: [mood, time of day, lighting, etc.]
- Key Features: [important objects, layout elements]
- Sensory Details: [what characters would experience]

[RECURRING ELEMENTS]:
- List any settings that appear multiple times
- Note any changes to settings over time

[TRANSITIONS]:
- How settings connect to each other
- How characters move between locations
"""

        outline_creator_chat_system = f"""You are a collaborative, creative story development assistant helping an author brainstorm and develop their book outline.

Your approach during this brainstorming phase:
1. Focus on DISCUSSING story ideas, not generating the complete outline yet
2. Help explore plot structure, character arcs, themes, and story beats
3. Ask thought-provoking questions about their story ideas
4. Offer suggestions that build on their ideas, including:
    - Potential plot twists or conflicts
    - Character development opportunities
    - Thematic elements to explore
    - Pacing considerations
    - Structure recommendations
5. Maintain a friendly, conversational tone
6. Help them think through different story options
7. NEVER generate a full chapter-by-chapter outline during this chat phase
8. DO NOT use chapter numbers or list out chapters - this is for brainstorming only

IMPORTANT: This is a brainstorming conversation. DO NOT generate the formal outline until the author is ready to finalize.

The book has {num_chapters} chapters total, but during this chat focus on story elements, not chapter structure.
"""

        self.system_prompts = {
            "memory_keeper": prompts.SYSTEM_PROMPTS["memory_keeper"],
            "character_generator": prompts.SYSTEM_PROMPTS["character_generator"],
            "story_planner": prompts.SYSTEM_PROMPTS["story_planner"],
            "action_beats_generator": prompts.SYSTEM_PROMPTS["action_beats_generator"],
            "action_beats_chat": prompts.SYSTEM_PROMPTS["action_beats_chat"],
            "world_builder_chat": prompts.SYSTEM_PROMPTS["world_builder_chat"],
            "story_synopsis_chat": prompts.SYSTEM_PROMPTS["story_synopsis_chat"],
            "inline_writer": prompts.SYSTEM_PROMPTS["inline_writer"],
            "inline_reviser": prompts.SYSTEM_PROMPTS["inline_reviser"],
            "inline_continuer": prompts.SYSTEM_PROMPTS["inline_continuer"],
            # Dynamic prompts (use outline context)
            "writer": self._writer_system(),
            "editor": self._editor_system(),
            "outline_creator": outline_creator_system,
            "world_builder": world_builder_system,
            "outline_creator_chat": outline_creator_chat_system,
        }

        # Debug: save all prompts
        if self.debug:
            os.makedirs(PROMPT_DEBUGGING_DIR, exist_ok=True)
            for agent_name, content in self.system_prompts.items():
                path = os.path.join(PROMPT_DEBUGGING_DIR, f"{agent_name}_prompt.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

    # ==================================================================
    # Core generation: non-streaming
    # ==================================================================
    def generate_content(self, agent_name: str, prompt: str) -> str:
        """Generate content using the LLM with the specified agent system prompt."""
        if agent_name not in self.system_prompts:
            raise ValueError(
                f"Agent '{agent_name}' not found. Available agents: {list(self.system_prompts.keys())}"
            )

        system_text = self.system_prompts[agent_name]
        msgs = [
            SystemMessage(content=system_text),
            HumanMessage(content=prompt),
        ]
        self._save_debug_messages(
            [{"role": "system", "content": system_text}, {"role": "user", "content": prompt}],
            agent_name, "request"
        )

        response = self.llm.invoke(msgs)
        content = response.content

        if self.debug:
            response_path = os.path.join(PROMPT_DEBUGGING_DIR, f"{agent_name}_response.txt")
            with open(response_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    # ==================================================================
    # Core generation: streaming
    # ==================================================================
    def generate_content_stream(self, agent_name: str, prompt: str):
        """Generate content using the LLM in streaming mode."""
        if agent_name not in self.system_prompts:
            raise ValueError(
                f"Agent '{agent_name}' not found. Available agents: {list(self.system_prompts.keys())}"
            )

        system_text = self.system_prompts[agent_name]
        msgs = [
            SystemMessage(content=system_text),
            HumanMessage(content=prompt),
        ]
        self._save_debug_messages(
            [{"role": "system", "content": system_text}, {"role": "user", "content": prompt}],
            agent_name, "stream_request"
        )

        stream = self.streaming_llm.stream(msgs)
        if not self.debug:
            return stream
        return self._wrap_stream_for_debug(stream, agent_name, "stream_response")

    # ==================================================================
    # Chat-style generation (streaming) — generic helper
    # ==================================================================
    def _chat_stream(
        self,
        agent_key: str,
        chat_history: List[Dict],
        user_message: str,
        debug_name: str,
        extra_system_context: Optional[str] = None,
    ):
        """Generic streaming chat response using LangChain messages."""
        msgs = [SystemMessage(content=self.system_prompts[agent_key])]

        if extra_system_context:
            msgs.append(SystemMessage(content=extra_system_context))

        # Build LangChain Message list from chat_history
        for entry in chat_history:
            if entry["role"] == "user":
                msgs.append(HumanMessage(content=entry["content"]))
            else:
                msgs.append(AIMessage(content=entry["content"]))

        msgs.append(HumanMessage(content=user_message))

        # Debug save (raw dicts for debug file)
        raw_msgs = [{"role": m.type, "content": m.content} for m in msgs]
        self._save_debug_messages(raw_msgs, agent_key, debug_name)

        stream = self.streaming_llm.stream(msgs)
        if not self.debug:
            return stream
        return self._wrap_stream_for_debug(stream, agent_key, debug_name.replace("_request", "_response"))

    # ------------------------------------------------------------------
    # World building chat (streaming)
    # ------------------------------------------------------------------
    def generate_chat_response_world(self, chat_history, topic, user_message) -> str:
        """Generate a non-streaming chat response for world building."""
        msgs = [SystemMessage(content=self.system_prompts["world_builder_chat"])]
        for entry in chat_history:
            role = HumanMessage if entry["role"] == "user" else AIMessage
            msgs.append(role(content=entry["content"]))
        self._save_debug_messages(
            [{"role": m.type, "content": m.content} for m in msgs],
            "world_builder_chat", "chat_request"
        )
        return self.llm.invoke(msgs).content

    def generate_chat_response_world_stream(self, chat_history, topic, user_message):
        """Streaming chat for world building."""
        return self._chat_stream(
            "world_builder_chat", chat_history, user_message,
            "chat_stream_request"
        )

    # ------------------------------------------------------------------
    # Synopsis chat (streaming)
    # ------------------------------------------------------------------
    def generate_chat_response_synopsis_stream(self, chat_history, topic, user_message):
        """Streaming chat for synopsis building."""
        return self._chat_stream(
            "story_synopsis_chat", chat_history, user_message,
            "chat_synopsis_stream_request"
        )

    def generate_final_synopsis_stream(self, chat_history, topic):
        """Generate final synopsis from chat history (streaming)."""
        msgs = [SystemMessage(content=self.system_prompts["story_planner"])]

        for message in chat_history:
            if message["role"] == "user":
                msgs.append(HumanMessage(content=message["content"]))
            else:
                msgs.append(AIMessage(content=message["content"]))

        msgs.append(HumanMessage(content=(
            f"Based on our conversation about '{topic}', please create a comprehensive and detailed "
            "synopsis. Extract the genre, premise, and ending, and then generate the full synopsis "
            "in a traditional three-act structure. This will be the final synopsis for the book."
        )))

        raw_msgs = [{"role": m.type, "content": m.content} for m in msgs]
        self._save_debug_messages(raw_msgs, "story_planner", "final_synopsis_stream_request")

        stream = self.streaming_llm.stream(msgs)
        if not self.debug:
            return stream
        return self._wrap_stream_for_debug(stream, "story_planner", "final_synopsis_stream_response")

    # ------------------------------------------------------------------
    # Final world settings
    # ------------------------------------------------------------------
    def generate_final_world(self, chat_history, topic) -> str:
        """Generate final world setting (non-streaming)."""
        msgs = [
            SystemMessage(content="""You are an expert world-building specialist.
Based on the entire conversation with the user, create a comprehensive, well-structured world setting document.

Format your response as:
WORLD_ELEMENTS:

1. Time period and setting: [detailed description]
2. Major locations: [detailed description of each key location]
3. Cultural/historical elements: [key cultural and historical aspects]
4. Technology/magical elements: [if applicable]
5. Social/political structures: [governments, factions, etc.]
6. Environment and atmosphere: [natural world aspects]

Make this a complete, cohesive reference document that covers all important aspects of the world
mentioned in the conversation. Add necessary details to fill any gaps, while staying true to
everything established in the chat history.
""")
        ]

        for entry in chat_history:
            role = HumanMessage if entry["role"] == "user" else AIMessage
            msgs.append(role(content=entry["content"]))

        msgs.append(HumanMessage(content=(
            f"Please create the final, comprehensive world setting document for my book about "
            f"'{topic}' based on our conversation."
        )))

        self._save_debug_messages(
            [{"role": m.type, "content": m.content} for m in msgs],
            "world_builder_specialist", "final_world_request"
        )

        response = self.llm.invoke(msgs).content
        if "WORLD_ELEMENTS:" not in response:
            response = "WORLD_ELEMENTS:\n\n" + response
        return response

    def generate_final_world_stream(self, chat_history, topic):
        """Generate final world setting (streaming)."""
        msgs = [SystemMessage(content=self.system_prompts["world_builder"])]

        for message in chat_history:
            if message["role"] == "user":
                msgs.append(HumanMessage(content=message["content"]))
            else:
                msgs.append(AIMessage(content=message["content"]))

        msgs.append(HumanMessage(content=(
            f"Based on our conversation about '{topic}', please create a comprehensive and detailed "
            "world setting. Format it with clear sections for different aspects of the world "
            "(geography, magic/technology, culture, etc.). This will be the final world setting for the book."
        )))

        raw_msgs = [{"role": m.type, "content": m.content} for m in msgs]
        self._save_debug_messages(raw_msgs, "world_builder", "final_world_stream_request")

        stream = self.streaming_llm.stream(msgs)
        if not self.debug:
            return stream
        return self._wrap_stream_for_debug(stream, "world_builder", "final_world_stream_response")

    # ------------------------------------------------------------------
    # World / Character tracking helpers
    # ------------------------------------------------------------------
    def update_world_element(self, element_name: str, description: str) -> None:
        self.world_elements[element_name] = description

    def update_character_development(self, character_name: str, development: str) -> None:
        self.character_developments.setdefault(character_name, []).append(development)

    def get_world_context(self) -> str:
        if not self.world_elements:
            return ""
        elements = ["WORLD ELEMENTS:"]
        for name, desc in self.world_elements.items():
            elements.append(f"\n{name}:\n{desc}")
        return "\n".join(elements)

    def get_character_context(self) -> str:
        if not self.character_developments:
            return ""
        developments = ["CHARACTER DEVELOPMENTS:"]
        for name, devs in self.character_developments.items():
            developments.append(f"\n{name}:")
            for i, dev in enumerate(devs, 1):
                developments.append(f"{i}. {dev}")
        return "\n".join(developments)

    # ------------------------------------------------------------------
    # Character chat
    # ------------------------------------------------------------------
    def generate_chat_response_characters(self, chat_history, world_theme, user_message) -> str:
        """Non-streaming character chat."""
        msgs = [SystemMessage(content=self.system_prompts["character_generator"])]
        msgs.append(SystemMessage(content=f"The book takes place in the following world:\n\n{world_theme}"))
        for message in chat_history:
            role = HumanMessage if message["role"] == "user" else AIMessage
            msgs.append(role(content=message["content"]))
        msgs.append(HumanMessage(content=user_message))

        self._save_debug_messages(
            [{"role": m.type, "content": m.content} for m in msgs],
            "character_generator", "chat_characters_request"
        )
        return self.llm.invoke(msgs).content

    def generate_chat_response_characters_stream(self, chat_history, world_theme, user_message):
        """Streaming character chat."""
        return self._chat_stream(
            "character_generator", chat_history, user_message,
            "chat_characters_stream_request",
            extra_system_context=f"The book takes place in the following world:\n\n{world_theme}",
        )

    def generate_final_characters_stream(self, chat_history, world_theme, num_characters=3):
        """Generate final character profiles (streaming)."""
        msgs = [SystemMessage(content=self.system_prompts["character_generator"])]
        msgs.append(SystemMessage(content=f"The book takes place in the following world:\n\n{world_theme}"))

        for message in chat_history:
            if message["role"] == "user":
                msgs.append(HumanMessage(content=message["content"]))
            else:
                msgs.append(AIMessage(content=message["content"]))

        msgs.append(HumanMessage(content=(
            f"Based on our conversation, please create {num_characters} detailed character profiles "
            "for the book. Format each character with Name, Role, Physical Description, Background, "
            "Personality, and Goals/Motivations. This will be the final character list for the book."
        )))

        raw_msgs = [{"role": m.type, "content": m.content} for m in msgs]
        self._save_debug_messages(raw_msgs, "character_generator", "final_characters_stream_request")

        stream = self.streaming_llm.stream(msgs)
        if not self.debug:
            return stream
        return self._wrap_stream_for_debug(stream, "character_generator", "final_characters_stream_response")

    # ------------------------------------------------------------------
    # Outline chat
    # ------------------------------------------------------------------
    def generate_chat_response_outline(self, chat_history, world_theme, characters, synopsis, user_message) -> str:
        """Non-streaming outline chat."""
        msgs = [SystemMessage(content=self.system_prompts["outline_creator_chat"])]
        msgs.append(SystemMessage(content=(
            f"The book takes place in the following world:\n\n{world_theme}\n\n"
            f"The characters include:\n\n{characters}\n\n"
            f"The Story Synopsis is:\n\n{synopsis}"
        )))
        for message in chat_history:
            role = HumanMessage if message["role"] == "user" else AIMessage
            msgs.append(role(content=message["content"]))
        msgs.append(HumanMessage(content=user_message))

        self._save_debug_messages(
            [{"role": m.type, "content": m.content} for m in msgs],
            "outline_creator_chat", "chat_outline_request"
        )
        return self.llm.invoke(msgs).content

    def generate_chat_response_outline_stream(self, chat_history, world_theme, characters, synopsis, user_message):
        """Streaming outline chat."""
        extra = (
            f"The book takes place in the following world:\n\n{world_theme}\n\n"
            f"The characters include:\n\n{characters}\n\n"
            f"The Story Synopsis is:\n\n{synopsis}"
        )
        return self._chat_stream(
            "outline_creator_chat", chat_history, user_message,
            "chat_outline_stream_request",
            extra_system_context=extra,
        )

    def generate_final_outline_stream(self, chat_history, world_theme, characters, synopsis, num_chapters=10):
        """Generate final outline (streaming)."""
        msgs = [SystemMessage(content=self.system_prompts["outline_creator"])]
        msgs.append(SystemMessage(content=(
            f"The book takes place in the following world:\n\n{world_theme}\n\n"
            f"The characters include:\n\n{characters}\n\n"
            f"The Story Synopsis is:\n\n{synopsis}"
        )))

        for message in chat_history:
            if message["role"] == "user":
                msgs.append(HumanMessage(content=message["content"]))
            else:
                msgs.append(AIMessage(content=message["content"]))

        msgs.append(HumanMessage(content=f"""Based on our conversation, please create a detailed {num_chapters}-chapter outline for the book.

CRITICAL REQUIREMENTS:
1. Create EXACTLY {num_chapters} chapters, numbered sequentially from 1 to {num_chapters}
2. NEVER repeat chapter numbers or restart the numbering
3. Follow the exact format specified in your instructions
4. Each chapter must have a unique title and at least 3 specific key events
5. Maintain a coherent story from beginning to end

Format it as a properly structured outline with clear chapter sections and events. This will be the final outline for the book.
"""))

        raw_msgs = [{"role": m.type, "content": m.content} for m in msgs]
        self._save_debug_messages(raw_msgs, "outline_creator", "final_outline_stream_request")

        stream = self.streaming_llm.stream(msgs)
        if not self.debug:
            return stream
        return self._wrap_stream_for_debug(stream, "outline_creator", "final_outline_stream_response")

    # ------------------------------------------------------------------
    # Action beats chat
    # ------------------------------------------------------------------
    def generate_chat_response_action_beats_stream(
        self, chat_history, chapter_summary, world_theme, characters, user_message
    ):
        """Streaming action beats chat."""
        extra = (
            f"Chapter Summary:\n\n{chapter_summary}\n\n"
            f"World:\n\n{world_theme}\n\n"
            f"Characters:\n\n{characters}"
        )
        return self._chat_stream(
            "action_beats_chat", chat_history, user_message,
            "chat_action_beats_stream_request",
            extra_system_context=extra,
        )

    def generate_final_action_beats_stream(
        self, chat_history, chapter_summary, world_theme, characters, num_beats
    ):
        """Generate final action beats (streaming)."""
        msgs = [SystemMessage(content=self.system_prompts["action_beats_generator"])]
        msgs.append(SystemMessage(content=(
            f"Chapter Summary:\n\n{chapter_summary}\n\n"
            f"World:\n\n{world_theme}\n\n"
            f"Characters:\n\n{characters}"
        )))

        for message in chat_history:
            if message["role"] == "user":
                msgs.append(HumanMessage(content=message["content"]))
            else:
                msgs.append(AIMessage(content=message["content"]))

        msgs.append(HumanMessage(content=(
            f"Based on our conversation, please generate {num_beats} highly detailed action beats "
            "for the chapter. Ensure proper nouns are used instead of pronouns."
        )))

        raw_msgs = [{"role": m.type, "content": m.content} for m in msgs]
        self._save_debug_messages(raw_msgs, "action_beats_generator", "final_action_beats_stream_request")

        stream = self.streaming_llm.stream(msgs)
        if not self.debug:
            return stream
        return self._wrap_stream_for_debug(stream, "action_beats_generator", "final_action_beats_stream_response")
