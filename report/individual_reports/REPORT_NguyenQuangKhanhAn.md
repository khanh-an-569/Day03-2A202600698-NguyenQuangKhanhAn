# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyễn Quang Khánh An
- **Student ID**: 2A202600698
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

- `src/agent/agent.py`: Implemented the ReAct loop (Thought → Action → Observation) with a focus on:
  - Parsing and recognizing `Action` statements (format `Action: tool_name(args)`) and detecting the `Final Answer`.
  - Executing tools via `_execute_tool()` supporting positional args, dict-style args, and simple string args.
  - Extracting JSON from the LLM responses and validating it using Pydantic before finalizing outputs.
- `src/tools/ecommerce.py`: Added Pydantic models `CompetitorItem` and `CompetitorReport` to standardize the agent's JSON output.
- `src/tools/web_search.py`: Lightweight search tool (DuckDuckGo HTML scraper) that returns structured snippets; includes a structured fallback when parsing fails.
- `src/core/provider_factory.py`: Provider factory function `get_provider_from_env` to construct `LocalProvider`/`OpenAIProvider`/`GeminiProvider` from `.env` configuration.
- `src/agent/run_ecommerce_agent.py`: Entrypoint to run the agent end‑to‑end with a provider from the factory.
- `src/chatbot.py`: Updated to run the baseline using `LocalProvider` and write outputs to the `logs/` directory.

---

## II. Debugging Case Study (10 Points)

**Problem description**
In some runs the agent returned `items: []` or otherwise invalid output despite calling `web_search`. The agent then reached `max_steps` and returned a "No results found" summary.

**Log evidence**

See the logs under `logs/`.

**Diagnosis**

- The HTML scraper in `src/tools/web_search.py` is brittle: DOM changes or regional blocking can cause parsing to fail and produce no usable results.
- The agent correctly followed the ReAct pattern but relied entirely on the tool output; when the tool returned a fallback with insufficient data, the agent had no alternative source and stopped early.

**Fix / Mitigation Implemented**

- Added a structured fallback in `web_search` that returns an observation describing the failure rather than raising an exception, allowing the agent to decide the next step.
- Added JSON extraction and validation in `src/agent/agent.py` (methods `_extract_json` and `_validate_output`) using the models from `src/tools/ecommerce.py`. Validation failures are logged under `AGENT_JSON_ERROR`.
- Updated the system prompt (`get_system_prompt()`) to instruct the LLM to avoid guessing, to use `null` for unknown fields, and to return JSON conforming to the schema for final outputs.

**Result**

After these changes, the agent returns either a validated JSON `CompetitorReport` or a clear diagnostic message indicating the search tool failure. This improves diagnosability and enables retry or fallback strategies.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Clearer reasoning**: ReAct forces the agent to state `Thought` and the next action explicitly, making the reasoning traceable.
2. **Reduced hallucinatio**: Factual information is obtained from deterministic tools instead of being invented by the LLM.
3. **Error modes shift to tools**: Failures are more often due to tools (scrapers, APIs), which can be mitigated with fallbacks and improved logging.
---

## IV. Future Improvements (5 Points)

- Replace the HTML scraper with an official search API (Bing, DuckDuckGo, Tavily API) to improve reliability.
- Add a `fetch_page(url)` tool to extract structured price and rating data from product pages.
- Normalize currencies and units, include a small mapping table and a currency-conversion tool when needed.
- Add caching and asynchronous tool calls to reduce latency and avoid duplicate requests.
- When Pydantic validation fails, send the invalid JSON back to the LLM for automated repair before rejecting the result.
---