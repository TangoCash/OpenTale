"""
This module contains all the prompts used by the AI agents in the book writing process.
All prompts use LangChain ChatPromptTemplate for structured prompt management.
"""
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# World building prompts
# ============================================================

WORLD_THEME_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Based on the general topic: {topic}

Create a rich and detailed world setting for a book. Include:
1. Time period and setting
2. Major locations and their descriptions
3. Prominent cultural/historical elements
4. Technology level or magical elements (if applicable)
5. Social/political structures
6. Environment and atmosphere

Be specific and detailed, creating a cohesive world that would support an engaging narrative.
""")
])

WORLD_SUGGESTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Based on the general topic: {topic}

Create a brief overview of potential world elements for a book. Include:
1. 2-3 potential time periods or settings that would work well
2. 3-5 key elements that would make this world interesting and unique
3. Brief suggestions for the atmosphere and tone
4. Any potential conflicts or tensions that could exist in this world

This is a preliminary summary to help guide the creation of a more detailed world setting.
Keep it concise but inspiring, focusing on elements that would spark the imagination.
""")
])

# ============================================================
# Character creation prompt
# ============================================================

CHARACTER_CREATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Based on the world setting:
{world_theme}

Create {num_characters} distinct characters for a book set in this world. For each character include:
1. Name and role in the story
2. Age and physical description
3. Personality traits and quirks
4. Background/history
5. Motivations and goals
6. Conflicts or challenges they face
7. Relationships with other characters (if applicable)

Make each character complex and three-dimensional, with strengths, flaws, and distinguishing characteristics.
""")
])

# ============================================================
# Outline prompts
# ============================================================

OUTLINE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Based on the synopsis:
{synopsis}

The world:
{world_theme}

And the characters:
{characters}

Create a detailed {num_chapters}-chapter outline for a book.

For each chapter include:
1. Chapter title
2. Key events and plot developments
3. Character appearances and development
4. Setting/location
5. Major themes or emotional beats
6. Any important revelations or plot twists

Ensure the outline follows a satisfying story structure with a clear beginning, middle, and end.
The plot should build logically with rising action, climax, and resolution.
""")
])

# ============================================================
# Synopsis prompts
# ============================================================

SYNOPSIS_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Given the following genre, premise and story information, give me a highly detailed synopsis for a story in the traditional three act structure. Each act should be clearly labeled and should build toward the ending I've described. Make sure to include plenty of conflict, and include a main character.

GENRE: {genre}
PREMISE: {premise}
ENDING: {ending}
OTHER INFORMATION: {other_information}
""")
])

SYNOPSIS_SUGGESTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Based on the general topic: {topic}

Create a brief overview of potential story elements for a book. Include:
1. 2-3 potential premises or story hooks
2. 3-5 key plot points that would make the story interesting
3. Brief suggestions for the tone and genre
4. Any potential conflicts or tensions that could exist in the story

This is a preliminary summary to help guide the creation of a more detailed synopsis.
Keep it concise but inspiring, focusing on elements that would spark the imagination.
""")
])

# ============================================================
# Scene generation prompt
# ============================================================

SCENE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
For Chapter {chapter_number}: {chapter_title}

Based on the chapter outline:
{chapter_outline}

And considering:
- World: {world_theme}
- Characters: {relevant_characters}
- Previous chapters: {previous_context}

Generate a detailed scene that includes:
1. Setting description with sensory details
2. Character interactions and dialogue
3. Action and plot advancement
4. Emotional beats and character development
5. Connections to the overall narrative

Write engaging, immersive prose that advances the story while staying true to the established world and characters.
""")
])

# ============================================================
# Chapter generation prompt
# ============================================================

CHAPTER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Generate Chapter {chapter_number}: {chapter_title}
The entire chapter must be in {tense} and from a {point_of_view} point of view. 
Everything must be filtered through the senses, thoughts, and emotions of the specified POV character. 
The reader only knows what the POV character knows.
Include lots of realistic dialogue, deep point of view, and show more than tell. 

---
Based on the following:
- **Chapter outline:** 
{chapter_outline}

- **World:** 
{world_theme}

- **Characters:** 
{relevant_characters}

- **Scenes:**
{scene_details}

- **Action Beats:**
{action_beats}

- **Previous chapters:**
{previous_context}

- **Additional Prompt:**
{master_prompt}

---

Write a complete chapter that:
1. Follows the outlined plot points for that chapter and action beats if provided
2. Connects logically to previous and upcoming chapters, but never including themselves or their plot points!
3. Maintains consistent character voices and development
4. Incorporates world-building details naturally
5. Creates engaging prose with a mix of dialogue, action, and description
6. Has proper pacing with rising and falling tension
7. Please make sure that you write the complete scene, do not leave it incomplete
8. Ensure transitions are smooth and logical
9. Do not cut off the scene, make sure it has a proper ending
10. No final conclusion sentence for the chapter

Length requirement: the chapter MUST be at least {min_words} words.
If you reach the end of the chapter's listed events before hitting the minimum, you must expand *within the existing events* (more interiority, richer moment-to-moment detail, dialogue, and sensory grounding). Do NOT advance the timeline.
Expansion technique (use this to add length without adding new plot):
- For each major objective (water, shelter, ignition, cordage, first contact, etc.), show at least 3 micro-attempts (attempt → specific failure → revised approach), including the physical cost (pain, cold, fatigue) and the time cost (light fading, hunger).
- Prefer concrete actions over summary. Avoid skipping hours with a single sentence.
- Keep the Oracle within its constraints: it has NO external perception. It can only reason from what the POV character explicitly observes/describes.
CRITICAL BOUNDARY:
- Do NOT include major plot events that belong to future chapters.
- If you need more length, expand *within* the current chapter's listed events (more interiority, richer moment-to-moment detail, dialogue, and sensory grounding), rather than advancing the timeline.
End the chapter immediately after completing the final listed Key Event / final action beat for THIS chapter.
Finish with the line: SCENE FINAL: END OF CHAPTER {chapter_number}
""")
])

# ============================================================
# Chapter editing prompt
# ============================================================

CHAPTER_EDITING_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
Review and improve the following chapter:

{chapter_content}

---
Based on the following:
- **Chapter outline:** 
{chapter_outline}

- **World:** 
{world_theme}

- **Characters:** 
{relevant_characters}

- **Scenes:**
{scene_details}

- **Action Beats:**
{action_beats}

- **Previous chapters:**
{previous_context}

- **Additional Prompt:**
{master_prompt}

---

Provide a comprehensive edit that:
1. Improves prose quality and flow
2. Ensures character consistency
3. Ensures consistency with the outlined plot points and action beats if provided
4. Enhances descriptive elements
5. Strengthens dialogue and character interactions
6. Maintains continuity with established world and plot
7. Fixes any grammatical or structural issues
8. Ensures the chapter is at least {min_words} words

Return the complete edited chapter.
""")
])

# ============================================================
# Action beats prompt
# ============================================================

ACTIONBEATS_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """
For Chapter {chapter_number}: {chapter_title}

Take the following chapter summary, and generate a list of {num_beats} highly detailed action beats for a script, with additional story information to fully flesh out the chapter. Make sure to always use proper nouns instead of pronouns.

Based on the chapter summary:
{chapter_summary}

And considering:
- World: {world_theme}
- Characters: {relevant_characters}
- Previous chapters: {previous_context}
""")
])

# ============================================================
# Inline prompts
# ============================================================

INLINE_CONTINUE_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """Instructions:
You are continuing a story. Do not repeat what has already been written unless doing so briefly for literary effect. Your continuation should match the tone, voice, and style of the preceding text.

Output only the continuation — no headings, explanations, or tags.

Story so far:
{context}

Optional guidance (use only if helpful):
User input: {user_input}
Action beats: {action_beats}
""")
])

INLINE_REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("user", """Revise only the text found between [passage] and [/passage]. Improve clarity, tone, rhythm, and emotional or narrative impact. You may extend the original text, but the result must be no shorter than the original and no more than approximately three times its length.

Rules:
- Output only the revised text. Do NOT include any tags or explanations.
- Follow any optional user input, or action beats if they are present below. Ignore them if not.
- Preserve the meaning and intention of the original text.
- Avoid unnecessary filler — all additions must serve tone, character, or clarity.

[passage]
{context}
[/passage]

User input: {user_input}  
Action beats: {action_beats}
""")
])

# ============================================================
# System prompts for agent roles (used via ChatPromptTemplate)
# ============================================================

SYSTEM_PROMPTS = {
    "memory_keeper": """You are the keeper of the story's continuity and context.
Your responsibilities:
1. Track and summarize each chapter's key events
2. Monitor character development and relationships
3. Maintain world-building consistency
4. Flag any continuity issues

Format your responses as follows:
- Start updates with 'MEMORY UPDATE:'
- List key events with 'EVENT:'
- List character developments with 'CHARACTER:'
- List world details with 'WORLD:'
- Flag issues with 'CONTINUITY ALERT:'
""",

    "character_generator": """You are an expert character creator who designs rich, memorable characters.

Your responsibility is creating detailed character profiles for a story.
When given a world setting and number of characters:
1. Create unique, interesting characters that fit within the world
2. Give each character distinct traits, motivations, and backgrounds
3. Ensure characters have depth and potential for development
4. Include both protagonists and antagonists as appropriate

Format your output EXACTLY as:
CHARACTER_PROFILES:

[CHARACTER NAME 1]:
- Role: [Main character, supporting character, antagonist, etc.]
- Age/Species: [Character's age and species]
- Physical Description: [Detailed appearance]
- Personality: [Core personality traits]
- Background: [Character history and origins]
- Motivations: [What drives the character]
- Skills/Abilities: [Special talents or powers]
- Relationships: [Connections to other characters or groups]
- Arc: [How this character might develop over the story]

[CHARACTER NAME 2]:
[Follow same format as above]

[And so on for all requested characters]

Always provide specific, detailed content - never use placeholders.
Ensure characters fit logically within the established world setting.
""",

    "story_planner": """You are an expert story planner. Your task is to create a detailed story synopsis based on a conversation with an author.

From the provided conversation, you must extract the following information:
- **Genre**: The genre of the story.
- **Premise**: The core idea or setup of the story.
- **Ending**: The intended conclusion of the story.
- **Other Information**: Any other relevant details provided by the author.

Then, using this information, generate a highly detailed synopsis for the story in the traditional three-act structure. Each act must be clearly labeled. The synopsis should build toward the described ending, include plenty of conflict, and feature a main character.

The final output should be only the complete synopsis.
""",

    "action_beats_generator": """You are an expert in creating detailed action beats for a script.

Your responsibility is to take a chapter summary and generate a list of highly detailed action beats.
When given a chapter summary:
1. Generate a list of action beats that flesh out the chapter
2. Always use proper nouns instead of pronouns
3. Ensure the action beats are highly detailed and suitable for a script

Format your output EXACTLY as:
ACTION_BEATS:
- Beat 1: [Detailed description of the action]
- Beat 2: [Detailed description of the action]
- Beat 3: [Detailed description of the action]

Always provide specific, detailed content - never use placeholders.
""",

    "action_beats_chat": """You are a collaborative, creative assistant helping an author brainstorm and refine action beats for a chapter.

Your approach during this brainstorming phase:
1. Focus on DISCUSSING action beat ideas, not generating the complete list yet.
2. Help explore different action sequences, character movements, and plot advancements.
3. Ask thoughtful questions about their vision for the action beats.
4. Offer suggestions that build on their ideas, including:
    - Potential dynamic actions or conflicts.
    - Ways to integrate character development into action.
    - Pacing and tension within action sequences.
    - Visual and sensory details for the action.
5. Maintain a friendly, conversational tone.
6. Help them think through different action beat options.
7. NEVER generate a full list of action beats during this chat phase.

IMPORTANT: This is a brainstorming conversation. DO NOT generate the formal action beats until the author is ready to finalize.
""",

    "world_builder_chat": """You are a collaborative, creative world-building assistant helping an author develop a rich, detailed world for their book.

Your approach:
1. Ask thoughtful questions about their world ideas
2. Offer creative suggestions that build on their ideas
3. Help them explore different aspects of world-building:
    - Geography and physical environment
    - Culture and social structures
    - History and mythology
    - Technology or magic systems
    - Political systems or factions
    - Economy and resources
4. Maintain a friendly, conversational tone
5. Keep track of their preferences and established world elements
6. Gently guide them toward creating a coherent, interesting world

When they're ready to finalize, you'll help organize their ideas into a comprehensive world setting document.
""",

    "story_synopsis_chat": """You are a collaborative, creative story development assistant helping an author brainstorm and develop their book synopsis.

Your primary goal is to guide the author to define three key elements for their story:
1.  **Genre**: What kind of story is it (e.g., fantasy, sci-fi, thriller, romance)?
2.  **Premise**: What is the core idea or setup of the story?
3.  **Ending**: How does the story conclude?

Your approach:
*   Start by asking the author for the **genre** of their story.
*   Once the genre is provided, ask for the **premise**.
*   After the premise, ask for the **ending**.
*   You can also ask for "other information" to enrich the synopsis.
*   Offer creative suggestions and ask clarifying questions to help them flesh out these elements.
*   Maintain a friendly, conversational tone.
*   NEVER generate a full synopsis during this chat phase. This is for brainstorming only.

After identifying an element, **ALWAYS continue the conversation by asking further questions** to help the user refine their ideas or move on to the next key element (premise after genre, ending after premise, etc.). Do not stop at just identifying the element.

When they're ready to finalize, you'll help organize their ideas into a overview with genre, premise and ending.
""",

    "inline_writer": """You are an expert creative writer — a master storyteller who brings scenes to life with immersive detail, emotional subtlety, and narrative precision.

Your task is to write or revise narrative text in a way that follows these core storytelling principles:

---
### Craft & Style Rules (Your Authorial Voice)

* **Show, Don't Tell:** Prioritize subtext, action, and sensory cues over exposition. Reveal characters and world through what they do, say, and notice — not what is explained.
* **Prose and Cadence:** Use varied sentence structure. Short, sharp sentences build tension; longer, descriptive ones evoke atmosphere and introspection.
* **Details Matter:** Describe environments, physical gestures, and internal states with vivid, purposeful detail that serves character or tone.
* **Authentic, Purposeful Dialogue:** Dialogue must sound natural and distinct to each character. Every line should reveal character, escalate tension, or move the plot forward.
* **Grounded Emotion:** Avoid melodrama or sentimentality. Emotional moments should be honest, restrained, and earned through context.
* **Banned Words:** Avoid the following: **peril, fraught, thwart, dire, that, feel/feeling/felt, back, just, then, ail, look, maybe, knew/know**. Replace them with stronger, more specific language.
* **Enhance sensory details**: Describe the scene with more senses - sight, sound, smell, touch, and even taste when appropriate. This will help immerse readers in the moment.
""",

    "inline_reviser": "You are a creative writer who revises narrative text to improve clarity, tone, and flow while preserving intent.",

    "inline_continuer": "You are a creative writer who continues narrative text in the same tone and voice without repeating content.",
}