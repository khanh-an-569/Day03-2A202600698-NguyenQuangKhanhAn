import ast
import json
import re
from datetime import datetime
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


def _extract_price(text: str) -> Optional[float]:
    match = re.search(r"(?:Price:\s*)?\$\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _extract_rating(text: str) -> Optional[float]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*out of\s*5(?:\s*stars)?", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _extract_review_count(text: str) -> Optional[int]:
    match = re.search(r"([0-9][0-9,]*)\s*ratings?", text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


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
        if not payload.get("timestamp"):
            payload["timestamp"] = datetime.utcnow().isoformat()
        model = self.output_schema.model_validate(payload)
        return model.model_dump_json(indent=2, ensure_ascii=False)

    def _enrich_payload(
        self,
        payload: Dict[str, Any],
        *,
        step_count: int,
        latency_ms: int,
        observations: List[str],
    ) -> Dict[str, Any]:
        enriched = dict(payload)
        enriched.setdefault("timestamp", datetime.utcnow().isoformat())
        enriched.setdefault("model", self.llm.model_name)
        provider_name = getattr(self.llm, "provider_name", None)
        if not provider_name:
            provider_name = self.llm.__class__.__name__.replace("Provider", "").lower()
        enriched.setdefault("provider", provider_name)
        enriched.setdefault("steps", step_count)
        enriched.setdefault("latency_ms", latency_ms)
        enriched.setdefault("tool_calls", len(observations))
        enriched.setdefault("observations_count", len(observations))
        return enriched

    def _build_fallback_report(
        self,
        user_input: str,
        observations: List[str],
        *,
        step_count: int,
        latency_ms: int,
    ) -> str:
        items: List[Dict[str, Any]] = []

        for observation_text in observations:
            try:
                parsed_observation = json.loads(observation_text)
            except Exception:
                continue

            if not isinstance(parsed_observation, list):
                continue

            for result in parsed_observation:
                if not isinstance(result, dict):
                    continue

                title = str(result.get("title") or "").strip()
                snippet = str(result.get("snippet") or "").strip()
                source = str(result.get("source") or "unknown").strip()
                url = result.get("url")

                text_blob = f"{title} {snippet}"
                price = _extract_price(text_blob)
                rating = _extract_rating(text_blob)
                review_count = _extract_review_count(text_blob)

                if not title or title.lower().startswith("no results"):
                    continue

                notes_parts = []
                if snippet:
                    notes_parts.append(snippet)
                if price is not None:
                    notes_parts.append(f"price={price}")
                if rating is not None:
                    notes_parts.append(f"rating={rating}")
                if review_count is not None:
                    notes_parts.append(f"reviews={review_count}")

                items.append(
                    {
                        "product_name": title,
                        "source": source,
                        "price": price,
                        "rating": rating,
                        "review_count": review_count,
                        "url": url,
                        "notes": "; ".join(notes_parts) if notes_parts else None,
                    }
                )

        if items:
            summary = f"Found {len(items)} competitor result(s) for {user_input}."
        else:
            summary = f"No reliably parsed competitor results found for {user_input}."

        fallback_payload = {
            "query": user_input,
            "items": items,
            "summary": summary,
        }

        fallback_payload = self._enrich_payload(
            fallback_payload,
            step_count=step_count,
            latency_ms=latency_ms,
            observations=observations,
        )

        validated = self.output_schema.model_validate(fallback_payload)
        return validated.model_dump_json(indent=2, ensure_ascii=False)

    def run(self, user_input: str) -> str:
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        # Scratchpad accumulates the full conversation context fed to the LLM each step
        scratchpad: List[str] = [f"User input: {user_input}"]
        current_prompt = user_input
        last_response = ""
        observations: List[str] = []
        step_latencies: List[int] = []

        for step in range(1, self.max_steps + 1):
            result = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())
            last_response = (result.get("content") or "").strip()
            step_latency_ms = int(result.get("latency_ms") or 0)
            step_latencies.append(step_latency_ms)

            if not last_response:
                logger.log_event(
                    "AGENT_EMPTY_RESPONSE",
                    {"step": step}
                )

                scratchpad.append(
                    "[System] The model returned an empty response. "
                    "Please continue and output either Action or Final Answer."
                )
                current_prompt = "\n\n".join(scratchpad)
                continue

            # Track performance metrics if usage data is available (OpenAI/Gemini providers)
            if result.get("usage"):
                tracker.track_request(
                    provider=result.get("provider", self.llm.model_name),
                    model=self.llm.model_name,
                    usage=result["usage"],
                    latency_ms=step_latency_ms,
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
                    payload = json.loads(_extract_json(final_answer) or "{}")
                    payload = self._enrich_payload(
                        payload,
                        step_count=step,
                        latency_ms=sum(step_latencies),
                        observations=observations,
                    )
                    validated = self.output_schema.model_validate(payload)
                    validated = validated.model_dump_json(indent=2, ensure_ascii=False)
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
                observations.append(observation)
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
        try:
            fallback = self._build_fallback_report(
                user_input,
                observations,
                step_count=self.max_steps,
                latency_ms=sum(step_latencies),
            )
            logger.log_event(
                "AGENT_FALLBACK",
                {"steps": self.max_steps, "status": "compiled_from_observations", "items": len(observations)},
            )
            return fallback
        except Exception as exc:
            logger.log_event(
                "AGENT_FALLBACK_ERROR",
                {"steps": self.max_steps, "error": str(exc), "last_response": last_response[:300]},
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