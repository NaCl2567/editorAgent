import json
import operator
import os
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from file import read_file, write_file
from search import duckduckgo_search, tavily_search

# from langchain_ollama import ChatOllama
# llm = ChatOllama(
#     model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
#     base_url="http://localhost:11434",
# )

import os
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),   # 默认模型
    api_key=os.environ.get("DEEPSEEK_API_KEY"),                # 从环境变量读取密钥
    # base_url 默认为 https://api.deepseek.com，一般不用改
)

WORKDIR = Path.cwd()
max_iterations = 5

NOISE_LINE_PREFIXES = (
    "image",
    "images",
    "photo",
    "photos",
    "figure",
    "fig.",
    "advertisement",
    "cookie",
    "privacy",
    "terms",
    "sign in",
    "log in",
    "subscribe",
    "menu",
)

IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp")


def clean_search_text(
    text: str, max_sentences: int = 4, max_chars: int = 600
) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    text = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    text = re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?document[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`~]+", " ", text)

    cleaned_lines = []
    seen_lines = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" -|>*#\t")
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith(("url:", "href:", "source:")):
            continue
        if lowered.startswith(NOISE_LINE_PREFIXES):
            continue
        if any(ext in lowered for ext in IMAGE_FILE_EXTENSIONS) and len(line.split()) <= 12:
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue

        normalized = line.lower()
        if normalized in seen_lines:
            continue
        seen_lines.add(normalized)
        cleaned_lines.append(line)

    merged = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
    if not merged:
        return ""

    sentences = re.split(r"(?<=[.!?\u3002\uFF01\uFF1F])\s+", merged)
    summary_parts = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        next_length = current_length + len(sentence) + (1 if summary_parts else 0)
        if summary_parts and (
            len(summary_parts) >= max_sentences or next_length > max_chars
        ):
            break
        if not summary_parts and len(sentence) > max_chars:
            summary_parts.append(sentence[: max_chars - 3].rstrip() + "...")
            break
        summary_parts.append(sentence)
        current_length = next_length

    summary = " ".join(summary_parts).strip()
    if not summary:
        summary = merged[:max_chars].rstrip()
    if len(summary) < len(merged) and not summary.endswith(
        ("...", "\u3002", ".", "!", "?", "\uFF01", "\uFF1F")
    ):
        summary += "..."
    return summary


def format_search_documents(search_docs: List[dict], max_results: int = 3) -> str:
    formatted_results = []
    for index, doc in enumerate(search_docs[:max_results], start=1):
        title = clean_search_text(doc.get("title", ""), max_sentences=1, max_chars=120)
        url = (doc.get("url", "") or "").strip()
        summary = clean_search_text(doc.get("content", ""), max_sentences=4, max_chars=500)

        if not title and not summary:
            continue

        parts = [f"[{index}]"]
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        if summary:
            parts.append(f"Summary: {summary}")
        formatted_results.append("\n".join(parts))

    return "\n\n---\n\n".join(formatted_results)


def summarize_search_results(raw_text: str, max_results: int = 3) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    block_pattern = re.compile(
        r"\[(\d+)\]\s*\nTitle:\s*(.*?)\nURL:\s*(.*?)\nSummary:\s*(.*?)(?=\n\s*\[\d+\]\s*\nTitle:|\Z)",
        re.DOTALL,
    )
    matches = list(block_pattern.finditer(raw_text))
    if matches:
        formatted_results = []
        for index, match in enumerate(matches[:max_results], start=1):
            title = clean_search_text(match.group(2), max_sentences=1, max_chars=120)
            url = match.group(3).strip()
            summary = clean_search_text(match.group(4), max_sentences=4, max_chars=500)
            if not title and not summary:
                continue

            parts = [f"[{index}]"]
            if title:
                parts.append(f"Title: {title}")
            if url:
                parts.append(f"URL: {url}")
            if summary:
                parts.append(f"Summary: {summary}")
            formatted_results.append("\n".join(parts))
        if formatted_results:
            return "\n\n---\n\n".join(formatted_results)

    sections = re.split(r"\n\s*---+\s*\n", raw_text)
    formatted_sections = []
    for index, section in enumerate(sections[:max_results], start=1):
        summary = clean_search_text(section, max_sentences=4, max_chars=500)
        if summary:
            formatted_sections.append(f"[{index}]\nSummary: {summary}")

    if formatted_sections:
        return "\n\n---\n\n".join(formatted_sections)
    return clean_search_text(raw_text, max_sentences=6, max_chars=1200)
# TODO: 实现外部持久化记忆,分为执行历史中的有用信息（比如human——feedback重要，checknode给出的feedback次要）、以及用户可编辑的gold rules 以及 其他小的提示
# TODO: 写入前进行审阅，可以拒绝改写
# TODO：不只是完全覆盖？
# TODO: 多Agent每个负责一个部分

def merge_string_dicts(
    left: Optional[Dict[str, str]], right: Optional[Dict[str, str]]
) -> Dict[str, str]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def merge_history_dicts(
    left: Optional[Dict[str, List[str]]], right: Optional[Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    merged = {key: list(value) for key, value in (left or {}).items()}
    for key, value in (right or {}).items():
        merged.setdefault(key, [])
        merged[key].extend(value or [])
    return merged


def unique_list(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_heading_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("#", " ")).strip().lower()


def build_section_id(value: str, index: int) -> str:
    base = re.sub(r"[^\w]+", "-", normalize_heading_text(value), flags=re.UNICODE)
    base = base.strip("-_")
    return base or f"section-{index}"


def normalize_output_path(file_path: str, task: str) -> str:
    raw_value = (file_path or "").strip().replace("\\", "/")
    filename = Path(raw_value).name if raw_value else ""

    if not filename:
        task_lower = task.lower()
        if "readme" in task_lower:
            filename = "README.md"
        elif "review" in task_lower:
            filename = "literature_review.md"
        elif "summary" in task_lower:
            filename = "summary.md"
        else:
            filename = "report.md"

    return f"./output/{filename}"


def normalize_sections(raw_sections: List["SectionPlan"], task: str) -> List[dict]:
    normalized = []
    seen_ids = set()

    if not raw_sections:
        raw_sections = [
            SectionPlan(
                section_id="main",
                title="Main Section",
                description=f"Primary content for: {task}",
                writing_guidance="Write the complete document content in this section.",
                depends_on=[],
            )
        ]

    temp_rows = []
    for index, section in enumerate(raw_sections, start=1):
        section_id = build_section_id(section.section_id or section.title, index)
        while section_id in seen_ids:
            section_id = f"{section_id}-{index}"
        seen_ids.add(section_id)
        temp_rows.append(
            {
                "raw_id": (section.section_id or "").strip(),
                "raw_title": section.title.strip(),
                "section_id": section_id,
                "title": section.title.strip() or f"Section {index}",
                "description": section.description.strip(),
                "writing_guidance": section.writing_guidance.strip(),
                "depends_on": section.depends_on or [],
            }
        )

    alias_map = {}
    for row in temp_rows:
        alias_map[row["raw_id"]] = row["section_id"]
        alias_map[normalize_heading_text(row["raw_title"])] = row["section_id"]
        alias_map[row["section_id"]] = row["section_id"]

    for row in temp_rows:
        normalized_dependencies = []
        for dependency in row["depends_on"]:
            normalized_dependency = alias_map.get(
                dependency.strip(), alias_map.get(normalize_heading_text(dependency.strip()))
            )
            if normalized_dependency and normalized_dependency != row["section_id"]:
                normalized_dependencies.append(normalized_dependency)
        normalized.append(
            {
                "section_id": row["section_id"],
                "title": row["title"],
                "description": row["description"],
                "writing_guidance": row["writing_guidance"],
                "depends_on": unique_list(normalized_dependencies),
            }
        )

    return normalized


def extract_section_contents(content: str, sections: List[dict]) -> Dict[str, str]:
    content = (content or "").strip()
    if not content or not sections:
        return {}

    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
    title_to_id = {
        normalize_heading_text(section["title"]): section["section_id"]
        for section in sections
    }

    matches = []
    for match in heading_pattern.finditer(content):
        section_id = title_to_id.get(normalize_heading_text(match.group(2)))
        if section_id:
            matches.append((match.start(), section_id))

    if not matches:
        return {sections[0]["section_id"]: content}

    section_contents = {}
    for index, (start, section_id) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(content)
        chunk = content[start:end].strip()
        if chunk:
            section_contents[section_id] = chunk

    preamble = content[: matches[0][0]].strip()
    if preamble:
        first_section_id = sections[0]["section_id"]
        existing = section_contents.get(first_section_id, "")
        section_contents[first_section_id] = (
            f"{preamble}\n\n{existing}".strip() if existing else preamble
        )

    return section_contents


def ensure_section_heading(content: str, title: str) -> str:
    content = (content or "").strip()
    if not content:
        return f"## {title}"
    if re.match(r"^\s*#{1,6}\s+", content):
        return content
    return f"## {title}\n\n{content}".strip()


def serialize_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


class AgentState(TypedDict, total=False):
    task: str
    human_feedback: str

    plan: str
    file_path: str
    need_web_search: bool
    search_query: str
    document_format: str

    search_results: str
    check_search_results: str

    current_content: str

    sections: List[dict]
    sections_to_write: List[str]
    section_feedback: Dict[str, str]
    section_contents: Annotated[Dict[str, str], merge_string_dicts]
    section_edit_history: Annotated[Dict[str, List[str]], merge_history_dicts]
    current_section: dict

    check_result: str
    passed: bool

    iteration: int

    context: Annotated[List[BaseMessage], operator.add]
    logs: Annotated[List[str], operator.add]


class SectionPlan(BaseModel):
    section_id: str = Field(
        description="Stable lowercase identifier for the section."
    )
    title: str = Field(description="Display title for the section.")
    description: str = Field(description="What this section should cover.")
    writing_guidance: str = Field(
        description="Concrete writing guidance for the agent responsible for this section."
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Other section ids that this section should stay consistent with.",
    )


class PlanOutput(BaseModel):
    plan: str = Field(
        description="Actionable plan for the next writing round."
    )
    file_path: str = Field(
        description="Target relative file path under ./output/."
    )
    need_web_search: bool = Field(
        description="Whether the next writing round needs web search."
    )
    document_format: str = Field(
        description="Target article format, such as markdown report, memo, or blog post."
    )
    sections: List[SectionPlan] = Field(
        default_factory=list,
        description="Ordered section plan for the document.",
    )


class SearchQuery(BaseModel):
    search_query: str = Field(default="", description="Search query for retrieval.")


class CheckSearchQuery(BaseModel):
    search_query: str = Field(
        default="",
        description="Search query used to verify the current document content.",
    )


class SectionDraftOutput(BaseModel):
    summary: str = Field(description="Short summary of what changed in this section.")
    content: str = Field(
        description="Full markdown content for this section only."
    )


class SectionReview(BaseModel):
    section_id: str = Field(description="Section id being reviewed.")
    needs_revision: bool = Field(description="Whether this section needs revision.")
    advice: str = Field(description="Specific revision advice for this section.")
    related_sections: List[str] = Field(
        default_factory=list,
        description="Other section ids that should be updated together for consistency.",
    )
    reason: str = Field(
        default="",
        description="Why this section passed or failed.",
    )


class ReviewOutput(BaseModel):
    passed: bool = Field(description="Whether the whole document passes this review round.")
    reason: str = Field(description="High-level reason for the review result.")
    next_advice: str = Field(
        description="Overall advice for the next writing round."
    )
    section_reviews: List[SectionReview] = Field(
        default_factory=list,
        description="Per-section review results.",
    )


def llm_planner_node(state: AgentState) -> AgentState:
    task = state["task"]
    human_feedback = state.get("human_feedback", "")
    current_content = state.get("current_content", "")
    check_result = state.get("check_result", "")
    prior_sections = state.get("sections", [])
    section_feedback = state.get("section_feedback", {})
    section_edit_history = state.get("section_edit_history", {})

    system_prompt = """
You are the planner for a section-based document writing workflow.

Your job:
1. Understand the user's writing task.
2. Decide the target document format.
3. Split the document into practical sections.
4. Produce an actionable next-round plan.
5. Decide whether web search is necessary.

Rules:
- The file path must stay under ./output/.
- Prefer a stable section structure across iterations unless there is a strong reason to change it.
- Each section must have a title, what it should cover, and concrete writing guidance.
- If the reviewer already identified section-level issues, keep those sections or related sections in the plan.
- Keep the number of sections small and useful.
- Return structured output only.
"""

    user_prompt = f"""
Task:
{task}

Human feedback:
{human_feedback}

Current document content:
{current_content}

Previous review result:
{check_result}

Previous sections:
{serialize_json(prior_sections)}

Latest section feedback:
{serialize_json(section_feedback)}

Section edit history:
{serialize_json(section_edit_history)}
"""

    structured_llm = llm.with_structured_output(PlanOutput)
    response = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    normalized_sections = normalize_sections(response.sections, task)
    available_section_ids = [section["section_id"] for section in normalized_sections]

    requested_targets = [
        section_id
        for section_id in state.get("sections_to_write", [])
        if section_id in available_section_ids
    ]
    existing_section_contents = state.get("section_contents", {})
    for section_id in available_section_ids:
        if section_id not in existing_section_contents and section_id not in requested_targets:
            requested_targets.append(section_id)
    if not requested_targets:
        requested_targets = available_section_ids

    return {
        "plan": response.plan.strip() or f"Write the planned sections for: {task}",
        "file_path": normalize_output_path(response.file_path, task),
        "need_web_search": bool(response.need_web_search),
        "document_format": response.document_format.strip() or "markdown article",
        "sections": normalized_sections,
        "sections_to_write": unique_list(requested_targets),
        "logs": [
            f"[planner] format={response.document_format.strip() or 'markdown article'} "
            f"sections={available_section_ids}"
        ],
        "context": [
            HumanMessage(content=user_prompt),
            AIMessage(content=response.model_dump_json(indent=2, ensure_ascii=False)),
        ],
    }


def human_feedback_node(state: AgentState) -> AgentState:
    return {}


def _fallback_check_query(content: str) -> str:
    parts = []
    for line in content.splitlines():
        stripped = line.strip().lstrip("#*-0123456789. ").strip()
        if stripped:
            parts.append(stripped)
        if len(parts) >= 3 or len(" ".join(parts)) >= 160:
            break
    return " ".join(parts)[:160]


def build_check_search_query(content: str) -> str:
    content = content.strip()
    if not content:
        return ""

    fallback_query = _fallback_check_query(content)
    structured_llm = llm.with_structured_output(CheckSearchQuery)

    try:
        response = structured_llm.invoke(
            [
                SystemMessage(
                    content="""
Generate one short search query that can validate whether the current document
matches real-world sources and common document patterns.
Return only the query.
"""
                ),
                HumanMessage(
                    content=f"Current document content:\n{content[:4000]}\n\nGenerate 1 search query."
                ),
            ]
        )
        return (response.search_query or fallback_query).strip()
    except Exception:
        return fallback_query


def run_check_search(content: str) -> Tuple[str, str]:
    query = build_check_search_query(content)
    if not query:
        return "", ""
    return query, summarize_search_results(tavily_search(query, max_results=3))


search_instructions = SystemMessage(
    content="""
You will receive the full conversation history.

Generate one high-value search query that helps the next writing round the most.
Focus on the missing facts, structure references, or examples that are most useful now.
Return only the query.
"""
)


def web_search_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(SearchQuery)
    history = state.get("context", [])
    search_query = structured_llm.invoke([search_instructions] + history)
    query_text = search_query.search_query

    try:
        tavily_client = TavilySearch(max_results=3)
        data = tavily_client.invoke({"query": query_text})
        search_docs = data.get("results", data)
        formatted_search_docs = format_search_documents(search_docs, max_results=3)
    except Exception:
        formatted_search_docs = summarize_search_results(
            duckduckgo_search(query_text, max_results=3),
            max_results=3,
        )

    return {
        "search_query": query_text,
        "search_results": formatted_search_docs,
        "logs": [f"[web_search] query={query_text}"],
        "context": [
            AIMessage(
                content=(
                    f"Web Search Query:\n{query_text}\n\n"
                    f"Web Search Results:\n{formatted_search_docs}"
                )
            )
        ],
    }


def prepare_section_writes_node(state: AgentState) -> AgentState:
    sections = state.get("sections", [])
    section_contents = dict(state.get("section_contents", {}))
    extracted_sections = extract_section_contents(state.get("current_content", ""), sections)

    for section_id, section_content in extracted_sections.items():
        section_contents.setdefault(section_id, section_content)

    valid_section_ids = {section["section_id"] for section in sections}
    sections_to_write = [
        section_id
        for section_id in state.get("sections_to_write", [])
        if section_id in valid_section_ids
    ]
    if not sections_to_write:
        sections_to_write = [section["section_id"] for section in sections]

    return {
        "section_contents": section_contents,
        "sections_to_write": unique_list(sections_to_write),
        "logs": [f"[prepare_sections] targets={unique_list(sections_to_write)}"],
    }


def dispatch_section_writers(state: AgentState) -> List[Send]:
    section_lookup = {
        section["section_id"]: section for section in state.get("sections", [])
    }
    target_sections = state.get("sections_to_write", []) or list(section_lookup.keys())

    sends = []
    for section_id in target_sections:
        section = section_lookup.get(section_id)
        if not section:
            continue
        payload = dict(state)
        payload["current_section"] = section
        sends.append(Send("section_writer", payload))

    return sends


def section_writer_node(state: AgentState) -> AgentState:
    task = state["task"]
    plan = state.get("plan", "")
    section = state["current_section"]
    section_id = section["section_id"]
    section_title = section["title"]
    section_contents = state.get("section_contents", {})
    section_feedback = state.get("section_feedback", {})
    section_history = state.get("section_edit_history", {}).get(section_id, [])
    related_context = []
    for dependency_id in section.get("depends_on", []):
        dependency_content = section_contents.get(dependency_id, "").strip()
        if dependency_content:
            related_context.append(
                f"[{dependency_id}]\n{dependency_content}"
            )

    system_prompt = """
You are one writer agent responsible for exactly one document section.

Requirements:
- Write or revise only the assigned section.
- Respect the overall document format and the global plan.
- Use the section-specific review advice when provided.
- Keep this section consistent with related sections and prior edits.
- Preserve valid content and make the minimum necessary changes when revising.
- Return the full markdown for this section only.
"""

    user_prompt = f"""
Task:
{task}

Document format:
{state.get('document_format', '')}

Global plan:
{plan}

Assigned section:
{serialize_json(section)}

Current content of this section:
{section_contents.get(section_id, '')}

Relevant related sections:
{chr(10).join(related_context)}

Latest review advice for this section:
{section_feedback.get(section_id, '')}

Section edit history:
{serialize_json(section_history)}

Human feedback:
{state.get('human_feedback', '')}

Useful search results:
{state.get('search_results', '')}
"""

    structured_llm = llm.with_structured_output(SectionDraftOutput)
    response = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            *state.get("context", []),
            HumanMessage(content=user_prompt),
        ]
    )

    section_content = ensure_section_heading(response.content, section_title)
    section_summary = response.summary.strip() or f"Updated {section_title}"

    return {
        "section_contents": {section_id: section_content},
        "section_edit_history": {section_id: [f"writer: {section_summary}"]},
        "logs": [f"[section_writer:{section_id}] {section_summary}"],
        "context": [
            AIMessage(
                content=(
                    f"Section {section_id} summary:\n{section_summary}\n\n"
                    f"Section content:\n{section_content}"
                )
            )
        ],
    }


def assemble_document_node(state: AgentState) -> AgentState:
    sections = state.get("sections", [])
    section_contents = state.get("section_contents", {})

    ordered_sections = []
    for section in sections:
        section_id = section["section_id"]
        section_title = section["title"]
        ordered_sections.append(
            ensure_section_heading(section_contents.get(section_id, ""), section_title)
        )

    document_content = "\n\n".join(part for part in ordered_sections if part.strip()).strip()
    if not document_content:
        document_content = state.get("current_content", "")

    write_file(state["file_path"], document_content)

    return {
        "current_content": document_content,
        "logs": [f"[assemble_document] wrote {len(ordered_sections)} sections"],
        "context": [
            AIMessage(
                content=f"Assembled document content:\n{document_content}"
            )
        ],
    }


def read_and_check_node(state: AgentState) -> AgentState:
    file_path = state["file_path"]
    sections = state.get("sections", [])
    content = read_file(file_path)
    iteration = state.get("iteration", 0)
    check_query, check_search_results = run_check_search(content)

    system_prompt = """
You are the reviewer for a section-based document writing workflow.

Review inputs:
- current document content
- web evidence, if any
- section plan
- section modification history

Tasks:
1. Decide whether the whole document passes this round.
2. Review every section.
3. Identify which sections need revision.
4. Identify linked sections that should be updated together for consistency.
5. Give concrete advice per section.

Rules:
- passed=true only when no section needs revision.
- related_sections should only include section ids from the provided section plan.
- Keep advice concrete and actionable.
- Return structured output only.
"""

    user_prompt = f"""
Sections:
{serialize_json(sections)}

Section edit history:
{serialize_json(state.get('section_edit_history', {}))}

Current document content:
{content}

Validation search query:
{check_query}

Validation search results:
{check_search_results}
"""

    structured_llm = llm.with_structured_output(ReviewOutput)
    response = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    valid_section_ids = {section["section_id"] for section in sections}
    section_feedback = {}
    sections_to_write = []
    review_history = {}
    section_review_lines = []

    for review in response.section_reviews:
        section_id = review.section_id.strip()
        if section_id not in valid_section_ids:
            continue

        linked_sections = [
            related_id
            for related_id in review.related_sections
            if related_id in valid_section_ids and related_id != section_id
        ]
        advice_text = review.advice.strip() or review.reason.strip()
        if linked_sections:
            advice_text = (
                f"{advice_text}\nKeep in sync with: {', '.join(linked_sections)}".strip()
            )

        if review.needs_revision:
            section_feedback[section_id] = advice_text
            sections_to_write.append(section_id)
            sections_to_write.extend(linked_sections)
            review_history.setdefault(section_id, []).append(
                f"review: {advice_text or response.next_advice}"
            )
            for related_id in linked_sections:
                review_history.setdefault(related_id, []).append(
                    f"linked review from {section_id}: {advice_text or response.next_advice}"
                )

        section_review_lines.append(
            "\n".join(
                [
                    f"- {section_id}: needs_revision={str(review.needs_revision).lower()}",
                    f"  reason: {review.reason.strip()}",
                    f"  advice: {advice_text}",
                ]
            )
        )

    sections_to_write = unique_list(sections_to_write)
    passed = bool(response.passed) and not sections_to_write
    if not passed and not sections_to_write:
        sections_to_write = [section["section_id"] for section in sections]

    check_text = (
        f"PASSED: {str(passed).lower()}\n"
        f"REASON:\n{response.reason.strip()}\n\n"
        f"NEXT_ADVICE:\n{response.next_advice.strip()}\n\n"
        f"SECTION_ADVICE:\n" + ("\n".join(section_review_lines) if section_review_lines else "No section advice.")
    )

    return {
        "current_content": content,
        "check_search_results": check_search_results,
        "check_result": check_text,
        "passed": passed,
        "iteration": iteration + 1,
        "section_feedback": section_feedback,
        "sections_to_write": sections_to_write,
        "section_edit_history": review_history,
        "logs": [f"[read_and_check] query={check_query} passed={passed}"],
        "context": [
            AIMessage(
                content=(
                    f"Check Search Query:\n{check_query}\n\n"
                    f"Check Search Results:\n{check_search_results}\n\n"
                    f"Review Result:\n{check_text}"
                )
            )
        ],
    }


def route_after_planner(state: AgentState) -> Literal["web_search", "prepare_section_writes"]:
    if state.get("need_web_search", False):
        return "web_search"
    return "prepare_section_writes"


def route_after_check(state: AgentState) -> Literal["llm_planner", "end"]:
    if state["passed"]:
        return "end"

    if state["iteration"] >= max_iterations:
        return "end"

    return "llm_planner"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("llm_planner", llm_planner_node)
    graph.add_node("human_feedback", human_feedback_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("prepare_section_writes", prepare_section_writes_node)
    graph.add_node("section_writer", section_writer_node)
    graph.add_node("assemble_document", assemble_document_node)
    graph.add_node("read_and_check", read_and_check_node)

    graph.add_edge(START, "llm_planner")
    graph.add_edge("llm_planner", "human_feedback")

    graph.add_conditional_edges(
        "human_feedback",
        route_after_planner,
        {
            "web_search": "web_search",
            "prepare_section_writes": "prepare_section_writes",
        },
    )

    graph.add_edge("web_search", "prepare_section_writes")

    graph.add_conditional_edges(
        "prepare_section_writes",
        dispatch_section_writers,
        ["section_writer"],
    )

    graph.add_edge("section_writer", "assemble_document")
    graph.add_edge("assemble_document", "read_and_check")

    graph.add_conditional_edges(
        "read_and_check",
        route_after_check,
        {
            "llm_planner": "llm_planner",
            "end": END,
        },
    )

    return graph.compile(interrupt_before=["human_feedback"])


graph = build_graph()
