# EditorAgent

An AI-powered multi-agent document writing system built on [LangGraph](https://www.langchain.com/langgraph). It orchestrates planning, web research, parallel section writing, assembly, and automated review to produce well-structured documents from a single task description.

## Features

- **Multi-Agent Workflow** — Planner, Web Searcher, Section Writers, Assembler, and Reviewer agents collaborate in a loop
- **Parallel Section Writing** — Document sections are drafted concurrently via LangGraph's `Send` API for speed
- **Human-in-the-Loop** — Pauses after the planning stage so you can review and approve the document structure before writing begins
- **Dual Search Engine** — Primary search via [Tavily](https://tavily.com/) with full webpage fetching, with [DuckDuckGo](https://duckduckgo.com/) as automatic fallback
- **Automated Quality Check** — The Reviewer agent validates the assembled document against fresh web searches and provides section-level revision feedback
- **Section Dependency Tracking** — Sections can declare dependencies on other sections for consistency (e.g., a "Conclusion" section depends on "Findings")
- **Iterative Refinement** — Up to 5 planning-writing-review loops until the document passes quality checks

## Architecture

```
START
  └─> llm_planner ──> human_feedback (INTERRUPT)
         ├─> web_search ──> prepare_section_writes
         │                      └─> [section_writer × N] (parallel)
         │                              └─> assemble_document
         │                                      └─> read_and_check
         │                                              ├─> llm_planner (revise)
         │                                              └─> END
         └─> prepare_section_writes (skip search)
```

## Environment Setup

### Prerequisites

- Python 3.11+
- API keys for:
  - [DeepSeek](https://platform.deepseek.com/) (LLM)
  - [Tavily](https://tavily.com/) (Web Search)

### 1. Clone the repository

```bash
git clone https://github.com/NaCl2567/editorAgent.git
cd editorAgent
```

### 2. Configure environment variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

Then edit `.env` with your actual keys. The file looks like this:

```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
TAVILY_API_KEY=your_tavily_api_key_here
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | — | DeepSeek API key for the LLM |
| `DEEPSEEK_MODEL` | No | `deepseek-chat` | DeepSeek model to use |
| `TAVILY_API_KEY` | Yes | — | Tavily API key for web search |
| `LANGSMITH_API_KEY` | No | — | LangSmith API key for WebUI |
 

### 3. Install dependencies

```bash
conda create -n "your_virtual_env_name"
```


```bash
pip install langgraph langchain-deepseek langchain-tavily tavily duckduckgo-search httpx markdownify pydantic typing_extensions
```

Or use the LangGraph CLI, which reads dependencies from `langgraph.json`:

```bash
pip install langgraph-cli
```

## Running the Project

### Option A: LangGraph Dev Server (recommended)

This launches a local LangGraph API server with the graph accessible via a REST endpoint and a built-in playground UI:

```bash
pip install langgraph-cli
langgraph dev
```

The graph is exposed as configured in `langgraph.json` and can be invoked through the LangGraph API or the web playground.

### Option B: Direct Python Invocation

You can invoke the graph directly from Python. The compiled graph is exported at the module level:

```python
from graph import graph

# The graph pauses at human_feedback for approval
config = {"configurable": {"thread_id": "my-thread-1"}}

# Start the workflow
state = graph.invoke(
    {
        "task": "Write a comprehensive report about the Ottoman Empire",
        "context": [],
        "logs": [],
        "iteration": 0,
    },
    config=config,
)

# The graph interrupts here — check the plan, then resume
print(state["plan"])

# Resume with approval (empty feedback = approved)
state = graph.invoke(
    {"human_feedback": ""},  # empty string = approve; anything else = revision feedback
    config=config,
)

# The graph continues through: web_search -> section_writers -> assemble -> check
# If revisions are needed, it loops back; otherwise it ends
print(state.get("current_content"))
```

The graph interrupts **before** the `human_feedback` node on every iteration. To resume:
- Pass `{"human_feedback": ""}` to approve and continue
- Pass `{"human_feedback": "your revision notes"}` to send the plan back for rework

### Output

Generated documents are written to the `output/` directory (gitignored). The output path is automatically determined from the task content.

## Project Structure

```
editorAgent/
├── graph.py            # Main graph: state, nodes, routing, and graph construction
├── search.py           # Web search: Tavily (primary) and DuckDuckGo (fallback)
├── file.py             # Safe file I/O utilities (read_file / write_file)
├── langgraph.json      # LangGraph Cloud configuration
├── .env.example        # Environment variable template
├── .gitignore
└── output/             # Generated documents (gitignored)
```

| File | Role |
|------|------|
| `graph.py` | Defines the `AgentState` TypedDict, Pydantic output models, all LLM nodes (planner, writer, reviewer), routing logic, and the `StateGraph` construction. The compiled graph is exported as `graph`. |
| `search.py` | Implements `tavily_search()` which fetches full webpage content via `httpx` and converts to Markdown with `markdownify`, and `duckduckgo_search()` as a lightweight fallback. |
| `file.py` | Provides `read_file()` and `write_file()` with path safety guards (rejects absolute paths and directory traversal). |
| `langgraph.json` | Tells the LangGraph platform where to find the graph (`./graph.py:graph`), the env file, Python version, and dependencies. |

## How It Works

1. **Planning** — The planner agent receives the task and produces a structured plan with sections, a document format, and whether web search is needed.
2. **Human Approval** — The workflow pauses. You review the plan. Approve to continue, or provide feedback for revision.
3. **Web Search** (if needed) — Searches via Tavily (fetching full page content as Markdown) or falls back to DuckDuckGo. Results are cleaned and formatted.
4. **Parallel Writing** — Each section is assigned to an independent writer agent. Writers see their section's description, dependencies, and any existing content (for revision loops). All writers run concurrently.
5. **Assembly** — Section outputs are merged into a single Markdown document and saved to the `output/` directory.
6. **Review** — The reviewer agent reads the assembled document, runs validation web searches, and provides per-section feedback (pass/fail with notes).
7. **Loop or Finish** — If any section needs revision and the iteration count is under 5, the workflow loops back to planning with feedback. Otherwise it ends.
