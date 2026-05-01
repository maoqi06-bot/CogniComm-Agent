"""Prompt templates for long-term memory management."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


def build_memory_extraction_prompt(history_text: str, current_task: Optional[str] = None) -> str:
    """Build the prompt used to extract durable memories from a conversation."""
    task_line = f"\nCurrent task:\n{current_task}\n" if current_task else ""
    return f"""You are a long-term memory extraction specialist.
Extract only durable information that will help future agent sessions.

{task_line}
Conversation history:
{history_text}

Return a JSON array only. Do not include Markdown or prose.
Each item must use this schema:
{{
  "content": "A concise memory, 1-2 sentences.",
  "category": "user_preference | project_context | important_fact | working_state | skill_knowledge | conversation_summary",
  "importance_score": 0.0,
  "tags": ["short", "searchable", "labels"],
  "reason": "Why this memory is durable and useful."
}}

Save:
- Explicit user preferences, habits, and long-lived constraints.
- Project facts, architecture decisions, and current durable working state.
- Important technical facts or reusable solutions.

Do not save:
- Temporary tool failures, transient runtime errors, stack traces, cache paths, or debug noise.
- Vague summaries that would not change a future decision.
- Sensitive secrets or credentials.
"""


def build_memory_resolution_prompt(
    new_memory: Dict[str, Any],
    candidates: Iterable[Dict[str, Any]],
) -> str:
    """Build the prompt that decides how a new memory relates to existing memories."""
    payload = {
        "new_memory": new_memory,
        "similar_existing_memories": list(candidates),
    }
    return f"""You are the long-term memory curator for a coding agent.
Decide whether the new memory should be stored as new information, merged into an existing memory, replace an outdated memory, or be ignored.

Rules:
- Judge whether memories are the same topic before merging.
- If the new memory contradicts an old explicit user preference, the newest explicit preference wins.
- For project facts, keep the more specific, more recent, or better-supported fact.
- If the new memory only adds useful detail to an existing memory, update that existing memory.
- If it is a duplicate or low-value transient runtime/tool failure, ignore it.
- If it is merely semantically nearby but independent, create a new memory.
- Prefer preserving information over deletion when uncertain.

Return one JSON object only, with this schema:
{{
  "action": "create_new | update_existing | replace_existing | ignore",
  "target_memory_id": "existing id, required for update_existing/replace_existing/ignore duplicate",
  "updated_content": "final content for the target memory when updating or replacing",
  "category": "optional final category",
  "importance_score": 0.0,
  "tags": ["optional", "final", "tags"],
  "is_pinned": false,
  "reason": "short explanation"
}}

Input:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
"""


def build_memory_guidance_prompt() -> str:
    """Return the agent-facing guidance for using long-term memory tools."""
    return """

=== Long-Term Memory Guide ===
You can use these tools to manage durable memory:
- add_memory: save important user preferences, project constraints, decisions, and reusable solutions.
- search_memory: search relevant memories for context.
- update_memory: correct or refine existing memories.
- delete_memory: remove outdated or inaccurate memories.
- list_memories: inspect the memory store.
- get_memory_stats: inspect memory system statistics.

Save memories only when the information is durable and likely to affect future work. Do not save transient tool failures, temporary debug output, cache paths, stack traces, or vague observations.
"""


__all__ = [
    "build_memory_extraction_prompt",
    "build_memory_guidance_prompt",
    "build_memory_resolution_prompt",
]
