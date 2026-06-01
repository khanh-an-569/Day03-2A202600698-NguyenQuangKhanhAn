import ast
import json
import re
from typing import List, Dict, Any, Optional, Type

from pydantic import BaseModel

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from src.tools.ecommerce import CompetitorReport


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json(text: str) -> Optional[str]:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return cleaned[start : end + 1]


class ReActAgent:
    """
    ReAct-style Agent following the Thought → Action → Observation loop.

    Each step:
      1. Call LLM with the growing scratchpad as prompt.
      2. If LLM outputs "Final Answer:" → validate JSON against output_schema.
      3. If LLM outputs "Action: tool_name(args)" → execute tool, append
         "Observation: <result>" to scratchpad, loop.
      4. If max_steps reached → return last raw response.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 3,
        output_schema: Type[BaseModel] = CompetitorReport,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.output_schema = output_schema

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tools]
        )
        schema_example = json.dumps(
            {
                "query": "iPhone 15 case",
                "items": [
                    {
                        "product_name": "ESR Hybrid Case",
                        "source": "Amazon",
                        "price": 9.99,
                        "rating": 4.5,
                        "review_count": 1200,
                        "url": "https://amazon.com/...",
                        "notes": "Military-grade protection",
                    }
                ],
                "summary": "Found 1 competitor product. ESR leads with strong reviews.",
            },
            indent=2,
            ensure_ascii=False,
        )

        return f"""You are an e-commerce competitor and review analysis agent.

Available tools:
{tool_descriptions}

Rules:
- Follow the ReAct loop strictly: output exactly one of the two formats below each turn.
- Use tools to gather real data; never invent prices, ratings, or review counts.
- Set unknown numeric fields to null (not 0).
- Always call web_search at least once before writing the Final Answer.
- The Final Answer must be valid JSON matching the schema — no markdown fences, no extra keys.

Output format (choose ONE per response):

Option A — when you need a tool:
Thought: <your reasoning>
Action: tool_name("argument")

Option B — when you have enough data:
Thought: <your reasoning>
Final Answer: <raw JSON, no fences>

JSON schema example:
{schema_example}
""".strip()

    def _parse_action(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parse lines like:
          Action: web_search("iPhone 15 case")
          Action: web_search('iphone case', 3)
        Returns dict with tool_name and args, or None if not found.
        """
        # Primary: tool_name(args)
        match = re.search(
            r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)\s*$",
            text,
            re.DOTALL | re.MULTILINE,
        )
        if match:
            return {"tool_name": match.group(1).strip(), "args": match.group(2).strip()}

        # Fallback: tool_name "arg" (LLM forgot parentheses)
        match = re.search(
            r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+['\"](.+?)['\"]",
            text,
        )
        if match:
            return {"tool_name": match.group(1).strip(), "args": f'"{match.group(2).strip()}"'}

        return None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def _parse_tool_args(self, args: str) -> Any:
        """
        Convert the raw args string from Action: tool(args) into a Python value.
        Handles: JSON dict, JSON list, Python literal, bare string.
        Returns: dict → kwargs, tuple/list → positional args, scalar → single arg.
        """
        if not args:
            return ()

        candidate = args.strip()

        try:
            parsed = json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                parsed = candidate.strip("\"'")

        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
        return parsed

    def _validate_output(self, raw_text: str) -> str:
        """Extract JSON from raw_text, validate against output_schema, return pretty JSON."""
        json_text = _extract_json(raw_text)
        if json_text is None:
            raise ValueError(f"No JSON object found in: {raw_text[:200]!r}")

        payload = json.loads(json_text)
        model = self.output_schema.model_validate(payload)
        return model.model_dump_json(indent=2, ensure_ascii=False)

    def run(self, user_input: str) -> str:
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        # Scratchpad accumulates the full conversation context fed to the LLM each step
        scratchpad: List[str] = [f"User input: {user_input}"]
        current_prompt = user_input
        last_response = ""

        for step in range(1, self.max_steps + 1):
            if not response or not response.strip():
                logger.log_event(
                    "AGENT_EMPTY_RESPONSE",
                    {"step": step}
                )

                return {
                    "success": False,
                    "error": "LLM returned empty response"
                }

            # Track performance metrics if usage data is available (OpenAI/Gemini providers)
            if "usage" in result and result["usage"]:
                tracker.track_request(
                    provider=result.get("provider", self.llm.model_name),
                    model=self.llm.model_name,
                    usage=result["usage"],
                    latency_ms=result.get("latency_ms", 0),
                )

            logger.log_event(
                "AGENT_STEP",
                {
                    "step": step,
                    "latency_ms": result.get("latency_ms"),
                    "response": last_response,
                },
            )

            # --- Check for Final Answer first ---
            final_answer = self._parse_final_answer(last_response)
            if final_answer:
                try:
                    validated = self._validate_output(final_answer)
                    logger.log_event("AGENT_END", {"steps": step, "status": "validated"})
                    return validated
                except Exception as exc:
                    # JSON invalid — ask LLM to fix it rather than giving up
                    logger.log_event(
                        "AGENT_JSON_ERROR",
                        {"step": step, "error": str(exc), "raw": final_answer[:300]},
                    )
                    scratchpad.append(f"[System] JSON validation failed: {exc}. Retry.")
                    scratchpad.append(
                        f"Your last Final Answer had a JSON error: {exc}\n"
                        f"Raw output:\n{final_answer}\n\n"
                        "Please output a corrected Final Answer with valid JSON only (no fences)."
                    )
                    current_prompt = "\n\n".join(scratchpad)
                    continue

            # --- Check for Action ---
            action = self._parse_action(last_response)
            if action:
                observation = self._execute_tool(action["tool_name"], action["args"])
                logger.log_event(
                    "TOOL_CALL",
                    {
                        "step": step,
                        "tool": action["tool_name"],
                        "args": action["args"],
                        "observation": observation[:500],
                    },
                )
                # KEY: append both Thought/Action AND Observation so LLM "remembers" on next step
                scratchpad.append(last_response.strip())
                scratchpad.append(f"Observation: {observation}")
                current_prompt = "\n\n".join(scratchpad)
                continue

            # --- Neither Final Answer nor Action — nudge the LLM ---
            logger.log_event("AGENT_NO_ACTION", {"step": step, "response": last_response[:200]})
            scratchpad.append(last_response.strip())
            scratchpad.append(
                "[System] No Action or Final Answer detected. "
                "Please output either 'Action: tool_name(args)' or 'Final Answer: {json}'."
            )
            current_prompt = "\n\n".join(scratchpad)

        logger.log_event(
            "AGENT_END",
            {"steps": self.max_steps, "status": "max_steps_reached"},
        )
        return last_response or "No final answer produced."

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """Find tool by name and call it with parsed arguments."""
        for tool in self.tools:
            if tool["name"] == tool_name:
                func = tool.get("func")
                if not callable(func):
                    return f"[Error] Tool '{tool_name}' has no callable implementation."

                parsed_args = self._parse_tool_args(args)
                try:
                    if isinstance(parsed_args, dict):
                        result = func(**parsed_args)
                    elif isinstance(parsed_args, tuple):
                        result = func(*parsed_args)
                    elif parsed_args == ():
                        result = func()
                    else:
                        result = func(parsed_args)
                except TypeError as exc:
                    try:
                        result = func(str(parsed_args))
                    except Exception:
                        return f"[Error] Tool '{tool_name}' call failed: {exc}"

                if isinstance(result, (dict, list)):
                    return json.dumps(result, ensure_ascii=False, indent=2)
                return str(result)

        return f"[Error] Tool '{tool_name}' not found. Available: {[t['name'] for t in self.tools]}"