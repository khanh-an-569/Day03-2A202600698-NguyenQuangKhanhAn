# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: [Please fill your name]
- **Student ID**: [Please fill your ID]
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

Implemented and integrated the core components for the E-commerce Competitor / Review Analyzer Agent:

- `src/agent/agent.py`: Implemented a ReAct-style loop (Thought -> Action -> Observation) with:
  - robust Action parsing (`Action: tool_name(args)`) and Final Answer detection.
  - tool execution via `_execute_tool()` that supports positional, dict, and simple argument parsing.
  - JSON extraction and validation via Pydantic using the schema in `src/tools/ecommerce.py`.
- `src/tools/ecommerce.py`: Added `CompetitorItem` and `CompetitorReport` Pydantic models to enforce structured JSON output.
- `src/tools/web_search.py`: Added a lightweight DuckDuckGo HTML search scraper returning structured snippets for agent observations.
- `src/core/provider_factory.py`: Added provider factory `get_provider_from_env()` to initialize `LocalProvider`, `OpenAIProvider`, or `GeminiProvider` from `.env`.
- `src/agent/run_ecommerce_agent.py`: Added a runnable entrypoint that builds the agent from the provider factory and executes a query end-to-end.
- Updated `src/chatbot.py` to use `LocalProvider` for the chatbot baseline and save results to `logs/`.

Code highlights (references):
- ReAct loop and JSON validation: `src/agent/agent.py`
- Search tool and result normalization: `src/tools/web_search.py`
- Output schema: `src/tools/ecommerce.py`

Documentation added inline in the files and event logging was used extensively for traceability.

---

## II. Debugging Case Study (10 Points)

**Problem Description**

During initial runs the agent returned an empty `items` array despite issuing `web_search` calls (see logs under `logs/2026-06-01.log`). The agent then reached `max_steps` and returned a "No results found" summary.

**Log Evidence**

Excerpt from `logs/2026-06-01.log`:

```
{"event": "TOOL_CALL", "data": {"step": 1, "tool": "web_search", "args": "\"iphone 15 case\"", "observation": "[ { \"title\": \"No results parsed\", \"url\": null, ... } ]"}}
```

**Diagnosis**

- The DuckDuckGo HTML scraping in `src/tools/web_search.py` is brittle: HTML structure changes or regional blocking can lead to no parsed results.
- The agent correctly followed ReAct (issued `Action: web_search("...")`) but relied on the tool to produce usable structured data. When the tool returned the fallback "No results parsed" object, the agent had no further alternate source and stopped after `max_steps`.

**Fix / Mitigation Implemented**

- Added robust fallback in the tool: when parsing fails, return a clearly structured observation explaining the failure instead of raising an exception (so the agent can react). See `src/tools/web_search.py`.
- Implemented JSON extraction and strict validation in `src/agent/agent.py` (`_extract_json`, `_validate_output`) so that final outputs are validated against `CompetitorReport` and failures are logged as `AGENT_JSON_ERROR` for diagnosis.
- Updated system prompt in `get_system_prompt()` to instruct the LLM to use `null` for unknown fields and to avoid inventing data.

**Result**

After these changes the agent returns either a validated JSON `CompetitorReport` or a clear diagnostic message when the search tool fails. Example successful run (logged):

```
{"event":"AGENT_END","data":{"steps":1,"status":"validated"}}
```

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: The ReAct loop makes intermediate reasoning explicit (`Thought`) and separates deciding what to do (`Action`) from retrieving facts (`Observation`). This greatly reduces hallucination when external facts are required.
2. **Reliability**: A stand-alone chatbot may attempt to answer multi-step factual questions by speculating; the ReAct agent instead calls deterministic tools. That shifts error modes from hallucination to tool failures and parsing issues, which are easier to diagnose via logs.
3. **Observation Influence**: Feeding the `Observation` back into the prompt enables the agent to revise actions iteratively (for example, retrying with a different query). This made it straightforward to implement retries or to ask the LLM for normalized JSON if the first attempt failed.

---

## IV. Future Improvements (5 Points)

- Replace HTML scraping with a reliable Search API (Tavily, Bing, or DuckDuckGo API) to reduce parsing brittleness.
- Add a `fetch_page(url)` tool to extract structured price/rating from product pages for higher-fidelity data.
- Normalize currencies and prices with explicit parsing and use a small mapping table (e.g., handle $ / € / VND) and currency-conversion tool for consistent numeric analysis.
- Add asynchronous tool calls and caching to improve latency and avoid repeated requests for the same query.
- Add an automated JSON repair step: when Pydantic validation fails, send the invalid JSON back to the LLM and request a corrected JSON conforming to the schema.

---

**Appendix: How to run**

1. Ensure `.env` points to your model or API keys.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the agent with the provider chosen in `.env`:

```bash
python -m src.agent.run_ecommerce_agent "iPhone 15 case"
```

Replace the student metadata at the top and save as `REPORT_[YOUR_NAME].md` before submission.
