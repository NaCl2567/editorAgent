import operator
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch  # updated 1.0
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph

import os
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
    base_url="http://localhost:11434",
)

from pathlib import Path
WORKDIR = Path.cwd()

from search import duckduckgo_search, tavily_search
# TODO：实现联网check，重写prompt，editor用所有对话历史来作为上下文，checker仅使用联网搜索结果和当前文件内容
# TODO: 实现外部持久化记忆
max_iterations = 5

class AgentState(TypedDict):
    task: str
    file_path: str

    original_content: str
    current_content: str

    plan: str
    human_feedback: str

    need_web_search: bool
    search_query: str
    search_results: str

    edited_content: str
    check_result: str
    passed: bool

    iteration: int
    max_iterations: int

    logs: Annotated[list, operator.add]

class llmOutputInfoState(BaseModel):
    plan: str = Field(
        description=(
            "文件编辑计划。说明需要对目标文件做哪些修改、修改原因、预期结果，"
            "以及是否需要分步骤执行。"
        )
    )

    file_path: str = Field(
        description=(
            "目标文件路径。必须是相对于当前工作目录的路径，例如 './README.md'、"
            "'./src/main.py'。如果用户没有明确指定文件路径，则返回空字符串。"
        )
    )

    need_web_search: bool = Field(
        description=(
            "是否需要联网搜索。只有当任务需要最新信息、外部事实、API 文档、"
            "网页内容，或无法仅凭当前文件内容完成时，才返回 true。"
        )
    )

    search_query: str = Field(
        description=(
            "联网搜索关键词。仅当 need_web_search 为 true 时填写；"
            "否则返回空字符串。"
        )
    )

def llm_planner_node(state: AgentState) -> AgentState:
    task = state["task"]
    human_feedback = state.get("human_feedback", "")
    
    system_prompt = """
    你是一个文件编辑 agent 的规划器。
    你需要根据用户任务、当前文件内容、人工反馈和搜索结果，生成下一步编辑计划。

    你需要输出：
    1. 编辑计划
    2. 从task中提取输出文件路径目标文件路径（工作目录的相对路径file_path）.注意，只能写相对路径，前缀必须为"./"
    3. 是否需要联网搜索(need_web_search)
    4. 搜索查询语句（search_query）

    """

    user_prompt = f"""用户任务：{task}"""

    structured_llm = llm.with_structured_output(llmOutputInfoState)
    response = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    plan, file_path, need_web_search, search_query = response.plan, response.file_path, response.need_web_search, response.search_query


    return {
        **state,
        "plan": plan,
        "task" : task,
        "file_path": file_path,
        "need_web_search": need_web_search,
        "search_query": search_query,
        "logs": [f"[planner] {plan}"],
    }

def human_feedback_node(state: AgentState):
    pass

def web_search_node(state: AgentState) -> AgentState:
    query = state.get("search_query", "")

    if not query:
        return {
            **state,
            "search_results": "",
            "logs": state["logs"] + ["[web_search] skipped because query is empty"],
        }

    # results = duckduckgo_search(query, max_results=5)
    results = tavily_search(query, max_results=3)

    return {
        **state,
        "search_results": results,
        "logs": state["logs"] + [f"[web_search] query={query}"],
    }

def edit_tool_node(state: AgentState) -> AgentState:
    task = state["task"]
    current_content = state.get("current_content", "")
    plan = state["plan"]
    human_feedback = state.get("human_feedback", "")
    search_results = state.get("search_results", "")

    
    system_prompt = """
    你是一个严谨的文件编辑器。
    你需要根据用户任务、编辑计划、人工反馈和搜索结果，修改文件内容。

    要求：
    1. 只输出修改后的完整文件内容
    2. 不要输出解释
    3. 不要使用 Markdown 代码块包裹
    4. 保留无关内容
    5. 尽量做最小必要修改
    """

    user_prompt = """
    用户任务：
    {task}

    当前文件内容：
    {current_content}

    编辑计划：
    {plan}

    人工反馈：
    {human_feedback}

    搜索结果：
    {search_results}

    请输出修改后的完整文件内容。
    """

    system_prompt = system_prompt
    user_prompt = user_prompt.format(task=task, current_content=current_content, plan=plan, human_feedback=human_feedback, search_results=search_results)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    edited_content = response.content

    write_file(state["file_path"], edited_content)

    return {
        **state,
        "edited_content": edited_content,
        "current_content": edited_content,
        "logs": state["logs"] + ["[edit_tool] file edited"],
    }

def read_and_check_node(state: AgentState) -> AgentState:
    file_path = state["file_path"]
    content = read_file(file_path)
    # content = state["current_content"]

    task = state["task"]
    plan = state["plan"]
    iteration = state.get("iteration", 0)

    system_prompt = """
    你是一个文件编辑结果检查器。
    你需要判断当前文件是否已经满足用户任务。

    请严格输出以下格式：

    PASSED: true 或 false
    REASON:
    ...
    NEXT_ADVICE:
    ...
    """

    user_prompt = f"""
    用户任务：
    {task}

    编辑计划：
    {plan}

    当前文件内容：
    {content}

    当前迭代轮数：
    {iteration}
    最大迭代轮数：
    {max_iterations}

    请判断是否已经完成任务。
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    check_text = response.content
    passed = "passed: true" in check_text.lower()

    new_iteration = iteration + 1

    return {
        **state,
        "current_content": content,
        "check_result": check_text,
        "passed": passed,
        "iteration": new_iteration,
        "logs": state["logs"] + [f"[read_and_check] {check_text}"],
    }

def route_after_planner(state: AgentState) -> Literal["web_search", "edit_tool"]:
    if state.get("need_web_search", False):
        return "web_search"
    return "edit_tool"

def route_after_check(state: AgentState) -> Literal["llm_planner", "end"]:
    if state["passed"]:
        return "end"

    if state["iteration"] >= max_iterations:
        return "end"

    return "llm_planner"

def read_file(file_path: str, default: str = "") -> str:
    path = Path(file_path)

    if not path.exists():
        return default

    return path.read_text(encoding="utf-8")

def write_file(file_path: str, content: str) -> None:
    path = Path(file_path).expanduser()

    if path.is_absolute():
        raise ValueError(f"不允许写入绝对路径: {file_path}")

    full_path = (WORKDIR / path).resolve()

    if not str(full_path).startswith(str(WORKDIR.resolve())):
        raise ValueError(f"不允许写入工作目录之外的路径: {file_path}")

    full_path.parent.mkdir(parents=True, exist_ok=True)

    full_path.write_text(content, encoding="utf-8")


from pydantic import BaseModel, Field


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("llm_planner", llm_planner_node)
    graph.add_node("human_feedback", human_feedback_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("edit_tool", edit_tool_node)
    graph.add_node("read_and_check", read_and_check_node)

    graph.add_edge(START, "llm_planner")
    graph.add_edge("llm_planner", "human_feedback")

    graph.add_conditional_edges(
        "human_feedback",
        route_after_planner,
        {
            "web_search": "web_search",
            "edit_tool": "edit_tool",
        },
    )

    graph.add_edge("web_search", "edit_tool")
    graph.add_edge("edit_tool", "read_and_check")

    graph.add_conditional_edges(
        "read_and_check",
        route_after_check,
        {
            "llm_planner": "llm_planner",
            "end": END,
        },
    )

    return graph.compile(interrupt_before=['human_feedback'])

graph = build_graph()