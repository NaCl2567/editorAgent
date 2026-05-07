import operator
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch  # updated 1.0
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string, BaseMessage

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph

import os
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
    base_url="http://localhost:11434",
)

from pathlib import Path
from file import write_file, read_file
WORKDIR = Path.cwd()

from search import duckduckgo_search, tavily_search
# TODO：实现联网check，重写prompt，editor用所有对话历史来作为上下文，checker仅使用联网搜索结果(关注格式和主要内容的相关证据)和当前文件内容
# TODO: 实现外部持久化记忆
# TODO: 写入前进行审阅，可以拒绝改写
# TODO：不只是完全覆盖？
# TODO: 多Agent每个负责一个部分
max_iterations = 5

class AgentState(TypedDict):
    task: str

    human_feedback: str

    plan: str
    file_path: str
    need_web_search: bool
    search_query: str

    search_results: str

    current_content: str

    check_result: str
    passed: bool

    iteration: int

    context: Annotated[list[BaseMessage], operator.add]
    logs: Annotated[list[str], operator.add]

class planOutput(BaseModel):
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

def llm_planner_node(state: AgentState) -> AgentState:
    task = state["task"]
    human_feedback = state.get("human_feedback", "")
    current_content = state.get("current_content", "")
    check_result = state.get("check_result", "")
    
    system_prompt = """
    你是一个文件编辑 agent 的规划器，负责为后续编辑节点制定清晰、可执行的编辑计划。

    你的职责：
    1. 理解用户任务，判断目标文件类型和目标产物。
    2. 从用户任务中提取或推断输出文件路径 file_path。
    3. 根据任务要求、当前内容、人工反馈和上一轮检查建议，制定下一步编辑计划。
    4. 判断是否需要联网搜索 need_web_search。

    你不负责直接写最终正文，也不负责输出完整文件内容。

    路径规则：
    1. file_path 必须是工作目录下 output 文件夹中的相对路径。
    2. file_path 必须以 "./output/" 开头。
    3. 如果用户明确给出了文件名，例如 report.md，则转换为 "./output/report.md"。
    4. 如果用户给出了非 output 目录路径，例如 "./report.md" 或 "./docs/report.md"，必须改写到 "./output/" 下。
    5. 如果用户没有明确文件名，则根据任务类型生成合理文件名。
    例如：
    - 报告类任务使用 "./output/report.md"
    - README 类任务使用 "./output/README.md"
    - 论文综述类任务使用 "./output/literature_review.md"
    - 调研总结类任务使用 "./output/research_summary.md"
    - 代码文件根据语言使用合适后缀，例如 "./output/main.py"
    6. 不允许输出绝对路径。
    7. 不允许输出 "../"。
    8. 不允许输出不以 "./output/" 开头的路径。

    联网搜索判断规则：
    1. 如果任务需要最新信息、外部事实、论文、网页资料、API 文档、产品信息、统计数据、新闻、开源项目信息，则 need_web_search 为 true。
    2. 如果当前内容为空、很短、明显没有成型，且任务需要事实性内容支撑，则 need_web_search 为 true。
    3. 如果上一轮检查建议指出缺少事实依据、资料来源、关键背景、案例或最新信息，则 need_web_search 为 true。
    4. 如果任务只是润色、改写、排版、结构调整、语言优化，且当前内容已经足够支撑任务，则 need_web_search 为 false。
    5. 如果人工反馈明确要求不要联网，则 need_web_search 为 false。
    6. 如果人工反馈明确要求补充资料、查证、引用或搜索，则 need_web_search 为 true。

    计划要求：
    1. plan 应该是可执行的编辑计划，不要泛泛而谈。
    2. plan 需要说明文档结构或代码结构。
    3. plan 需要列出每个主要部分要写什么。
    4. 如果是对已有内容修改，plan 要说明保留哪些内容、改写哪些内容、补充哪些内容。
    5. 如果是根据上一轮检查建议返工，plan 要明确回应 check_result 中的问题。
    6. 如果需要联网搜索，plan 要说明需要搜索哪些信息，以及这些信息将用于哪些部分。
    7. 如果不需要联网搜索，plan 要说明将基于当前内容做哪些语言、结构或格式调整。

    请严格按照结构化输出 schema 返回，不要输出额外解释。
    """

    user_prompt = f"""
    用户任务：
    {task}

    人工反馈：
    {human_feedback}

    当前文件内容：
    {current_content}

    上一轮检查建议：
    {check_result}

    请生成下一步编辑计划、目标文件路径和是否需要联网搜索。
    """

    structured_llm = llm.with_structured_output(planOutput)
    response = structured_llm.invoke([
        SystemMessage(content=system_prompt.format(current_content=state.get("current_content", ""), check_result=state.get("check_result", ""))),
        HumanMessage(content=user_prompt),
    ])

    plan, file_path, need_web_search = response.plan, response.file_path, response.need_web_search


    return {
        **state,
        "plan": plan,
        "task" : task,
        "file_path": file_path,
        "need_web_search": need_web_search,
        "logs": [f"[planner] {plan}"],
        "context": [
            HumanMessage(content=user_prompt),
            AIMessage(content=response.model_dump_json(indent=2, ensure_ascii=False)),
        ],
    }

def human_feedback_node(state: AgentState):
    pass


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")
def web_search_node(state: AgentState):
    
    """ Retrieve docs from web search """

    # Search
    tavily_search = TavilySearch(max_results=3)

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['context'])
    
    # Search
    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {
        **state,
        "search_results": formatted_search_docs,
        "logs":  [f"[web_search] query={search_query}"],
    } 

# def web_search_node(state: AgentState) -> AgentState:
#     query = state.get("search_query", "")

#     if not query:
#         return {
#             **state,
#             "search_results": "",
#             "logs": ["[web_search] skipped because query is empty"],
#         }

#     # results = duckduckgo_search(query, max_results=5)
#     results = tavily_search(query, max_results=3)

#     return {
#         **state,
#         "search_results": results,
#         "logs":  [f"[web_search] query={query}"],
#     }

class EditPatchOutput(BaseModel):
    summary: str = Field(
        description="本次编辑的简要说明，说明修改了什么以及为什么修改。"
    )

    patch: str = Field(
        description=(
            "unified diff 格式的补丁内容。"
            "必须包含 --- 和 +++ 文件头，以及 @@ hunk。"
            "路径必须是相对于工作目录的路径，例如 ./README.md。"
        )
    )


class editOutput(BaseModel):
    summary: str
    content: str

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


    structured_llm = llm.with_structured_output(editOutput)


    response = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    current_content, edit_summary = response.content, response.summary

    write_file(state["file_path"], current_content)

    return {
        **state,
        "current_content": current_content,
        "logs":  ["[edit_tool] file edited"],
        "context": [AIMessage(f"Edit Summary: \n{edit_summary}\n\nCurrent File Content: \n{current_content}")]
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
        "logs":  [f"[read_and_check] {check_text}"],
        "context": [AIMessage(content=f"Recommendation: \n{check_text}\n\nPassed:{passed}\n")]
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



