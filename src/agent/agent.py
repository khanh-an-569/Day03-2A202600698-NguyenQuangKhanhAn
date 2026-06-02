import ast
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Type

from pydantic import BaseModel

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from src.tools.ecommerce import CompetitorReport, AgentMeta


# ─── JSON helpers ─────────────────────────────────────────────────────────────

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


# ─── Snippet parsers (fallback builder) ──────────────────────────────────────

def _extract_price(text: str) -> Optional[float]:
    m = re.search(r"(?:price:\s*)?\$\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _extract_rating(text: str) -> Optional[float]:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*out of\s*5(?:\s*stars)?", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _extract_review_count(text: str) -> Optional[int]:
    m = re.search(r"([0-9][0-9,]*)\s*ratings?", text, re.IGNORECASE)
    return int(m.group(1).replace(",", "")) if m else None


# ─── Console display helpers ──────────────────────────────────────────────────

_ICONS = {
    "AGENT_START":          "🚀",
    "AGENT_STEP":           "🧠",
    "TOOL_CALL":            "🔧",
    "AGENT_END":            "✅",
    "AGENT_FALLBACK":       "🔄",
    "AGENT_JSON_ERROR":     "⚠️ ",
    "AGENT_NO_ACTION":      "❓",
    "AGENT_EMPTY_RESPONSE": "💤",
    "LLM_METRIC":           "💰",
}

def _div(char: str = "─", w: int = 72) -> None:
    print(char * w)

def _print_event(event: str, data: Dict[str, Any]) -> None:
    icon = _ICONS.get(event, "•")

    if event == "AGENT_START":
        _div("═")
        print(f"{icon} AGENT START  │  query='{data['input']}'  model={data['model']}")
        _div()

    elif event == "AGENT_STEP":
        step = data["step"]
        ms   = data.get("latency_ms", "?")
        resp = data.get("response", "")
        print(f"\n{icon} Step {step}  ({ms} ms)")
        for line in resp.splitlines():
            line = line.strip()
            if not line:
                continue
            print(f"   {line}")
            if line.lower().startswith(("action:", "final answer:")):
                break

    elif event == "LLM_METRIC":
        tokens = data.get("total_tokens", "?")
        cost   = data.get("cost_estimate", 0)
        ms     = data.get("latency_ms", "?")
        print(f"   {icon} tokens={tokens}  cost=${cost:.5f}  latency={ms}ms")

    elif event == "TOOL_CALL":
        tool = data["tool"]
        args = str(data.get("args", ""))[:60]
        obs  = data.get("observation", "")
        # Count real results (have a url) vs stub errors
        try:
            parsed = json.loads(obs) if obs.strip().startswith("[") else []
            real   = [r for r in parsed if r.get("url")]
            count  = f"{len(real)} real result(s)" if real else "⚠ no results"
        except Exception:
            count = "?"
        print(f"   {icon} {tool}({args!r}…) → {count}")
        # Print actual URLs found
        try:
            for r in parsed:
                if r.get("url"):
                    print(f"      • {r['title'][:55]}  →  {r['url']}")
        except Exception:
            pass

    elif event == "AGENT_END":
        label = "✅ validated" if data.get("status") == "validated" else f"⚠ {data.get('status')}"
        print(f"\n{icon} DONE in {data.get('steps', '?')} step(s) — {label}")
        _div()

    elif event == "AGENT_FALLBACK":
        print(f"\n{icon} FALLBACK — compiled {data.get('items', 0)} item(s) from observations")
        _div()

    elif event == "AGENT_JSON_ERROR":
        print(f"\n⚠️  JSON error at step {data['step']}: {str(data['error'])[:80]}")

    elif event in ("AGENT_NO_ACTION", "AGENT_EMPTY_RESPONSE"):
        print(f"   {icon} {event} at step {data.get('step', '?')} — nudging LLM…")


# ─── Agent ────────────────────────────────────────────────────────────────────

class ReActAgent:
    """
    ReAct-style Agent following the Thought → Action → Observation loop.

    Key design decisions:
    - ONE action per LLM turn (prevents hallucinated multi-action chains).
    - Observations are always appended to the scratchpad so the LLM can read them.
    - search results include real URLs from OpenAI Responses API.
    - AgentMeta is attached AFTER Pydantic validation so it never breaks the schema.
    - Fallback builder parses raw observations when LLM never produces Final Answer.
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

    # ── System prompt ─────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self.tools
        )
        schema_example = json.dumps(
            {
                "query": "iPhone 15 case",
                "items": [
                    {
                        "product_name": "ESR Hybrid Case",
                        "source": "amazon.com",
                        "price": 9.99,
                        "rating": 4.5,
                        "review_count": 1200,
                        "url": "https://www.amazon.com/dp/B0CX...",
                        "notes": "Military-grade protection",
                    }
                ],
                "summary": "Found 1 product. ESR leads with 1.2K reviews at $9.99.",
            },
            indent=2,
            ensure_ascii=False,
        )

        return f"""You are an e-commerce competitor and review analysis agent.

Available tools:
{tool_descriptions}

Rules:
- ONE Action per response — never chain multiple actions in one reply.
- Use the URL and snippet from each search result to fill url, price, rating, review_count.
- If a field is unknown, use null — never invent numbers.
- The source field must be the website domain (e.g. "amazon.com", "bestbuy.com").
- Always call web_search at least once before the Final Answer.
- Final Answer must be raw JSON — no markdown fences, no extra keys outside the schema.

Output format (choose ONE per response):

Option A — need more data:
Thought: <reasoning>
Action: web_search("your query here")

Option B — ready to answer:
Thought: <reasoning>
Final Answer: <raw JSON>

JSON schema:
{schema_example}
""".strip()

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_action(self, text: str) -> Optional[Dict[str, str]]:
        """Parse the FIRST Action in the response (ignore any chained ones)."""
        m = re.search(
            r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)\s*$",
            text, re.DOTALL | re.MULTILINE,
        )
        if m:
            return {"tool_name": m.group(1).strip(), "args": m.group(2).strip()}

        # Fallback: LLM forgot parentheses — Action: web_search "query"
        m = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+['\"](.+?)['\"]", text)
        if m:
            return {"tool_name": m.group(1).strip(), "args": f'"{m.group(2).strip()}"'}

        return None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        m = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def _parse_tool_args(self, args: str) -> Any:
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

    # ── Output helpers ────────────────────────────────────────────────────────

    def _attach_meta(
        self,
        report_json: str,
        *,
        step_count: int,
        latency_ms: int,
        tool_calls: int,
    ) -> str:
        """
        Parse the validated JSON, attach AgentMeta, re-serialise.
        Meta is added AFTER Pydantic validation so it never breaks schema checks.
        """
        provider_name = getattr(self.llm, "provider_name", None) or \
            self.llm.__class__.__name__.replace("Provider", "").lower()

        payload = json.loads(report_json)
        payload["meta"] = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "model":      self.llm.model_name,
            "provider":   provider_name,
            "steps":      step_count,
            "latency_ms": latency_ms,
            "tool_calls": tool_calls,
        }
        # Re-validate to include meta in the Pydantic model
        final = self.output_schema.model_validate(payload)
        return final.model_dump_json(indent=2, ensure_ascii=False)

    def _validate_llm_output(self, raw_text: str) -> str:
        """
        Extract JSON from LLM final answer, validate against output_schema.
        Returns compact validated JSON (meta NOT attached here — caller does that).
        """
        json_text = _extract_json(raw_text)
        if json_text is None:
            raise ValueError(f"No JSON object found in: {raw_text[:200]!r}")
        payload = json.loads(json_text)

        # Strip any extra fields the LLM may have added (timestamp, model…)
        # by only keeping keys the schema knows about
        known = set(self.output_schema.model_fields.keys())
        payload = {k: v for k, v in payload.items() if k in known}

        model = self.output_schema.model_validate(payload)
        return model.model_dump_json(indent=2, ensure_ascii=False)

    def _build_fallback_report(self, user_input: str, observations: List[str]) -> str:
        """
        Build a CompetitorReport from raw web_search observations
        when the LLM never produced a valid Final Answer.
        Items are populated by parsing snippets for price/rating/review patterns.
        """
        items: List[Dict[str, Any]] = []

        for obs_text in observations:
            try:
                parsed_obs = json.loads(obs_text)
            except Exception:
                continue
            if not isinstance(parsed_obs, list):
                continue

            for result in parsed_obs:
                if not isinstance(result, dict):
                    continue

                title   = str(result.get("title")   or "").strip()
                snippet = str(result.get("snippet") or "").strip()
                url     = result.get("url")
                source  = str(result.get("source")  or "unknown").strip()

                # Skip stub / error entries
                if not title or title.lower().startswith(("search", "no result")):
                    continue

                blob         = f"{title} {snippet}"
                price        = _extract_price(blob)
                rating       = _extract_rating(blob)
                review_count = _extract_review_count(blob)

                note_parts = []
                if snippet:       note_parts.append(snippet[:200])
                if price is not None:        note_parts.append(f"price=${price}")
                if rating is not None:       note_parts.append(f"rating={rating}/5")
                if review_count is not None: note_parts.append(f"reviews={review_count:,}")

                items.append({
                    "product_name": title,
                    "source":       source,
                    "price":        price,
                    "rating":       rating,
                    "review_count": review_count,
                    "url":          url,
                    "notes":        "; ".join(note_parts) if note_parts else None,
                })

        summary = (
            f"Compiled {len(items)} result(s) for '{user_input}' from web search snippets."
            if items else
            f"Search returned no parseable results for '{user_input}'."
        )

        fallback_payload = {"query": user_input, "items": items, "summary": summary}
        validated = self.output_schema.model_validate(fallback_payload)
        return validated.model_dump_json(indent=2, ensure_ascii=False)

    # ── Core loop ─────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        start_data = {"input": user_input, "model": self.llm.model_name}
        logger.log_event("AGENT_START", start_data)
        _print_event("AGENT_START", start_data)

        scratchpad: List[str] = [f"User input: {user_input}"]
        current_prompt = user_input
        last_response  = ""
        observations: List[str] = []
        step_latencies: List[int] = []
        tool_call_count = 0

        for step in range(1, self.max_steps + 1):
            result        = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())
            last_response = (result.get("content") or "").strip()
            step_ms       = int(result.get("latency_ms") or 0)
            step_latencies.append(step_ms)

            # Empty response
            if not last_response:
                ed = {"step": step}
                logger.log_event("AGENT_EMPTY_RESPONSE", ed)
                _print_event("AGENT_EMPTY_RESPONSE", ed)
                scratchpad.append(
                    "[System] Empty response. Output either Action or Final Answer."
                )
                current_prompt = "\n\n".join(scratchpad)
                continue

            # Metrics
            if result.get("usage"):
                tracker.track_request(
                    provider=result.get("provider", self.llm.model_name),
                    model=self.llm.model_name,
                    usage=result["usage"],
                    latency_ms=step_ms,
                )
                _print_event("LLM_METRIC", {
                    "total_tokens": result["usage"].get("total_tokens"),
                    "cost_estimate": tracker.session_metrics[-1]["cost_estimate"],
                    "latency_ms":   step_ms,
                })

            step_data = {"step": step, "latency_ms": step_ms, "response": last_response}
            logger.log_event("AGENT_STEP", step_data)
            _print_event("AGENT_STEP", step_data)

            # ── Final Answer ──────────────────────────────────────────────────
            final_answer = self._parse_final_answer(last_response)
            if final_answer:
                try:
                    validated_json = self._validate_llm_output(final_answer)
                    # Attach metadata AFTER validation — this never breaks the schema
                    output = self._attach_meta(
                        validated_json,
                        step_count=step,
                        latency_ms=sum(step_latencies),
                        tool_calls=tool_call_count,
                    )
                    ed = {"steps": step, "status": "validated"}
                    logger.log_event("AGENT_END", ed)
                    _print_event("AGENT_END", ed)
                    return output
                except Exception as exc:
                    ed = {"step": step, "error": str(exc), "raw": final_answer[:300]}
                    logger.log_event("AGENT_JSON_ERROR", ed)
                    _print_event("AGENT_JSON_ERROR", ed)
                    scratchpad.append(f"[System] JSON validation failed: {exc}. Please fix.")
                    scratchpad.append(
                        f"Your Final Answer had a JSON error: {exc}\n"
                        f"Raw:\n{final_answer}\n\n"
                        "Output a corrected Final Answer (no fences, match the schema exactly)."
                    )
                    current_prompt = "\n\n".join(scratchpad)
                    continue

            # ── Action ───────────────────────────────────────────────────────
            action = self._parse_action(last_response)
            if action:
                tool_call_count += 1
                observation = self._execute_tool(action["tool_name"], action["args"])
                tool_data   = {
                    "step":        step,
                    "tool":        action["tool_name"],
                    "args":        action["args"],
                    "observation": observation,
                }
                logger.log_event("TOOL_CALL", {**tool_data, "observation": observation[:500]})
                _print_event("TOOL_CALL", tool_data)

                scratchpad.append(last_response.strip())
                scratchpad.append(f"Observation: {observation}")
                observations.append(observation)
                current_prompt = "\n\n".join(scratchpad)
                continue

            # ── No recognised output ─────────────────────────────────────────
            ed = {"step": step, "response": last_response[:200]}
            logger.log_event("AGENT_NO_ACTION", ed)
            _print_event("AGENT_NO_ACTION", ed)
            scratchpad.append(last_response.strip())
            scratchpad.append(
                "[System] No Action or Final Answer detected. "
                "Output 'Action: web_search(\"query\")' or 'Final Answer: {json}'."
            )
            current_prompt = "\n\n".join(scratchpad)

        # ── Max steps reached — build from raw observations ────────────────
        ed = {"steps": self.max_steps, "status": "max_steps_reached"}
        logger.log_event("AGENT_END", ed)
        _print_event("AGENT_END", ed)

        try:
            fallback_json = self._build_fallback_report(user_input, observations)
            output = self._attach_meta(
                fallback_json,
                step_count=self.max_steps,
                latency_ms=sum(step_latencies),
                tool_calls=tool_call_count,
            )
            fd = {"steps": self.max_steps, "status": "compiled_from_observations",
                  "items": tool_call_count}
            logger.log_event("AGENT_FALLBACK", fd)
            _print_event("AGENT_FALLBACK", fd)
            return output
        except Exception as exc:
            logger.log_event("AGENT_FALLBACK_ERROR",
                             {"error": str(exc), "last_response": last_response[:300]})
            return last_response or "No final answer produced."

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: str) -> str:
        for tool in self.tools:
            if tool["name"] != tool_name:
                continue
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

            return json.dumps(result, ensure_ascii=False, indent=2) \
                if isinstance(result, (dict, list)) else str(result)

        return f"[Error] Tool '{tool_name}' not found. Available: {[t['name'] for t in self.tools]}"