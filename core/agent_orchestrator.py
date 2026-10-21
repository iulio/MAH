"""
Agent Orchestrator - Planner -> Coder -> Verifier (+ self-healing fixer) loop.

Integrates:
- SandboxManager (local / Docker execution, DiffEngine-backed edits)
- GitGuard (automatic shadow checkpoint before every agent run, 1-click rollback)
- RepoMapper (<repo_map> injected into the planner/coder/verifier/ask system prompts)
- ContextCompactor (sliding-window compaction + token pre-flight before every NIM call)
- Hooks (pre/post tool validation and linting)
- MCP Bridge (external tool servers)
- Telemetry (NVIDIA NeMo Agent Toolkit compatible ATIF export)
- Sub-Agents (context-isolated specialists with parallel execution)

Modes:
1. ASK:   read-only codebase Q&A (list_files / read_file)
2. AGENT: plan, edit, verify, and fix in a bounded self-healing loop
"""

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.config import NIMClient, SettingsManager, get_settings_manager
from core.context_manager import ContextCompactor, create_context_compactor
from core.repo_map import RepoMapper, create_repo_mapper
from sandbox.executor import SandboxManager, SandboxType, TOOL_SCHEMAS, execute_tool_call
from sandbox.git_guard import CheckpointInfo, GitGuard

# New integrations
from core.hooks import HookedSandboxManager, create_hook_registry, get_default_hook_config
from core.mcp_bridge import initialize_mcp, shutdown_mcp, get_all_tools, MCPClientManager, ToolRegistry
from core.telemetry import TelemetryContext, ProfilerData, create_telemetry_context
from agents.subagents import SubAgentManager, SubAgentHarness, SubAgentResult

logger = logging.getLogger(__name__)

READ_ONLY_TOOLS = ("list_files", "read_file")
MUTATING_TOOLS = ("apply_edit", "create_file")
MAX_TOOL_ROUNDS = 25
MAX_CONTEXT_FILE_CHARS = 20000
RESPONSE_TOKEN_RESERVE = 2000


class AgentMode(Enum):
    """Execution mode for the agent."""
    ASK = "ask"          # Read-only Q&A
    AGENT = "agent"      # Full autonomous loop


@dataclass
class AgentMessage:
    """Represents a message in the agent conversation."""
    role: str  # user, assistant, tool, system
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentStep:
    """Represents a step in the agent execution."""
    step_type: str  # plan, edit, verify, fix
    model: str
    input_messages: List[AgentMessage]
    output: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class _PhaseResult:
    """Outcome of one model/tool loop (filled in as the loop generator runs)."""
    content: str = ""
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    last_exit_code: Optional[int] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)


class AgentHarness:
    """
    Multi-agent harness for code generation and editing.

    - Planner: analyzes the task and produces a step-by-step plan
    - Coder: implements the plan via apply_edit / create_file
    - Verifier: runs tests and reports results
    - Fixer: the coder again, fed with verification failures
    
    Integrations:
    - Hooks: pre/post tool validation and linting
    - MCP Bridge: external tool servers (GitHub, PostgreSQL, Context7, etc.)
    - Telemetry: NVIDIA NeMo Agent Toolkit compatible ATIF export
    - Sub-Agents: context-isolated specialists with parallel execution
    """

    def __init__(
        self,
        sandbox: SandboxManager,
        settings_manager: SettingsManager,
        mode: AgentMode = AgentMode.ASK,
        max_iterations: int = 5,
        git_guard: Optional[GitGuard] = None,
    ):
        self.sandbox = sandbox
        self.settings_manager = settings_manager
        self.mode = mode
        self.max_iterations = max_iterations
        self.git_guard = git_guard
        self._nim_client: Optional[NIMClient] = None
        self._conversation: List[AgentMessage] = []
        self._steps: List[AgentStep] = []
        self._repo_mapper: Optional[RepoMapper] = None
        self._repo_map_dirty = True
        self._last_checkpoint: Optional[CheckpointInfo] = None
        self.last_context_summary: Dict[str, Any] = {}
        
        # New integrations
        self._hook_registry = None
        self._hooked_sandbox = None
        self._mcp_manager: Optional[MCPClientManager] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._telemetry_context: Optional[TelemetryContext] = None
        self._sub_agent_manager: Optional[SubAgentManager] = None
        self._sub_agent_harness: Optional[SubAgentHarness] = None
        self._run_id = str(uuid.uuid4())
        self._run_start_time = time.time()

    # ==================== Lifecycle ====================

    async def __aenter__(self) -> "AgentHarness":
        self._nim_client = NIMClient(self.settings_manager.settings.nim)
        await self._nim_client.__aenter__()
        
        # Initialize hooks
        hooks_config = self.settings_manager.settings.hooks
        if hooks_config.enabled:
            hook_config = get_default_hook_config()
            hook_config.update({
                "bash_block_patterns": hooks_config.custom_patterns or hook_config["bash_block_patterns"],
                "lint_commands": {
                    "python": [hooks_config.linter, "check", "--output-format=concise"],
                    "fallback": ["flake8", "--select=E,W,F"]
                }
            })
            self._hook_registry = create_hook_registry(hook_config)
            self._hooked_sandbox = HookedSandboxManager(self.sandbox, hook_config)
        
        # Initialize MCP
        mcp_config = self.settings_manager.settings.mcp
        if mcp_config.servers:
            self._mcp_manager, self._tool_registry = await initialize_mcp(
                self.settings_manager.settings,
                TOOL_SCHEMAS
            )
        
        # Initialize telemetry
        telemetry_config = self.settings_manager.settings.telemetry
        if telemetry_config.enabled:
            self._telemetry_context = create_telemetry_context(settings=telemetry_config)
        
        # Initialize sub-agents
        self._sub_agent_manager = SubAgentManager(
            sandbox=self._hooked_sandbox or self.sandbox,
            settings_manager=settings_manager,
            repo_mapper=self._get_repo_mapper(),
            max_workers=2,
        )
        self._sub_agent_harness = SubAgentHarness(
            sandbox=self._hooked_sandbox or self.sandbox,
            settings_manager=settings_manager,
            max_iterations=max_iterations,
            git_guard=git_guard,
        )
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Export telemetry trajectory
        if self._telemetry_context and self._telemetry_context._prompt:
            try:
                await self._telemetry_context.export_trajectory_async()
            except Exception as e:
                logger.warning(f"Failed to export telemetry trajectory: {e}")
        
        # Shutdown MCP
        if self._mcp_manager:
            await shutdown_mcp()
        
        # Shutdown sub-agents
        if self._sub_agent_manager:
            self._sub_agent_manager.shutdown()
        
        if self._nim_client:
            await self._nim_client.__aexit__(exc_type, exc_val, exc_tb)
            self._nim_client = None

    # ==================== Settings helpers ====================

    def _get_model_for_role(self, role: str) -> str:
        models = self.settings_manager.settings.models
        return {
            "planner": models.planner_model,
            "coder": models.coder_model,
            "verifier": models.verifier_model,
        }.get(role, models.coder_model)

    def _get_tools_for_mode(self) -> List[Dict[str, Any]]:
        # Use merged tools from MCP registry if available
        if self._tool_registry:
            all_tools = self._tool_registry.all_tools
            if self.mode == AgentMode.ASK:
                return [t for t in all_tools if t.get("function", {}).get("name") in READ_ONLY_TOOLS]
            return all_tools
        
        # Fallback to local tools
        if self.mode == AgentMode.ASK:
            return self._read_only_tools()
        return TOOL_SCHEMAS

    @staticmethod
    def _read_only_tools() -> List[Dict[str, Any]]:
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in READ_ONLY_TOOLS]

    def _build_system_prompt(self, role: str) -> str:
        """Build the system prompt for a role (the repo map is attached by the context compactor)."""
        base_prompt = """You are an expert software engineer working in a multi-agent coding system.
You have access to a workspace sandbox with file operations and terminal execution.
Always be precise, concise, and follow best practices.
A <repo_map> of the codebase (file skeletons ranked by relevance) is provided; use it to navigate and
only read_file the files you actually need."""

        if role == "planner":
            return base_prompt + """

ROLE: PLANNER
Your job is to analyze the user's request and the workspace context, then create a detailed, step-by-step execution plan.

Guidelines:
- Break down complex tasks into small, atomic steps
- Identify which files need to be read, created, or modified
- Specify the exact changes needed for each file
- Consider edge cases and potential issues
- You may use list_files and read_file to inspect the codebase; do not modify anything
- Output a clear, numbered plan that the Coder can follow"""

        if role == "coder":
            return base_prompt + """

ROLE: CODER
Your job is to implement the plan by making precise file edits.

Guidelines:
- Use apply_edit for surgical changes (provide exact search_block and replace_block)
- Use create_file for new files
- Python edits are syntax-checked before saving; if an edit is rejected, fix it and retry
- Make minimal, focused changes
- Follow the existing code style and patterns
- If you need to see a file first, use read_file
- Always verify your changes make sense in context
- Output tool calls to execute the changes"""

        if role == "verifier":
            return base_prompt + """

ROLE: VERIFIER
Your job is to verify the implementation by running tests and checking for errors.

Guidelines:
- Run the appropriate test command (pytest, npm test, etc.)
- Check for linting/type errors (ruff, mypy, tsc, etc.)
- If tests fail, analyze the error output and provide specific feedback
- If all checks pass, confirm success
- Output tool calls to run verification commands"""

        return base_prompt

    # ==================== Repo map ====================

    def _get_repo_mapper(self) -> RepoMapper:
        if self._repo_mapper is None:
            cfg = self.settings_manager.settings.repo_map
            self._repo_mapper = create_repo_mapper(
                workspace_dir=str(self.sandbox.workspace_dir),
                token_budget=cfg.token_budget,
                include_patterns=cfg.include_patterns,
                exclude_patterns=cfg.exclude_patterns,
                max_file_size_kb=cfg.max_file_size_kb,
            )
        return self._repo_mapper

    def get_repo_map(self, query: str = "", force_refresh: bool = False, max_tokens: Optional[int] = None) -> str:
        """Return the formatted <repo_map> block (synchronous; used by the UI modal)."""
        mapper = self._get_repo_mapper()
        if force_refresh or self._repo_map_dirty:
            mapper.scan(force_refresh=True)
            self._repo_map_dirty = False
        return mapper.format_repo_map(query=query, max_tokens=max_tokens)

    def get_repo_map_stats(self) -> Dict[str, Any]:
        mapper = self._get_repo_mapper()
        if self._repo_map_dirty:
            mapper.scan(force_refresh=True)
            self._repo_map_dirty = False
        return mapper.get_stats()

    async def _build_repo_map(self, query: str) -> str:
        try:
            return await asyncio.to_thread(self.get_repo_map, query)
        except Exception as e:
            logger.warning("Repo map generation failed: %s", e)
            return ""

    # ==================== Git checkpoints (exposed to the UI) ====================

    def rollback_last_run(self) -> Dict[str, Any]:
        """1-click undo of the entire last agent run."""
        if not self.git_guard:
            return {"success": False, "error": "Git checkpoints are not enabled"}
        result = self.git_guard.rollback_last()
        if result.get("success"):
            self._repo_map_dirty = True
        return result

    def commit_accepted_changes(self, message: str) -> Dict[str, Any]:
        if not self.git_guard:
            return {"success": False, "error": "Git checkpoints are not enabled"}
        return self.git_guard.commit_accepted_changes(message)

    # ==================== Model I/O ====================

    async def _stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        if not self._nim_client:
            raise RuntimeError("NIM client not initialized; use 'async with AgentHarness(...)'")
        async for chunk in self._nim_client.chat_completion_stream(
            model=model, messages=messages, tools=tools,
            tool_choice=tool_choice, temperature=temperature,
        ):
            yield chunk

    @staticmethod
    def _parse_stream_chunk(chunk: str) -> Dict[str, Any]:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return {}

    def _read_context_files(self, context_files: Optional[List[str]]) -> str:
        context = ""
        for f in context_files or []:
            try:
                content = self.sandbox.read_file(f)
                if len(content) > MAX_CONTEXT_FILE_CHARS:
                    content = content[:MAX_CONTEXT_FILE_CHARS] + "\n... [truncated]"
                context += f"\n--- File: {f} ---\n{content}\n"
            except Exception as e:
                context += f"\n--- File: {f} (ERROR: {e}) ---\n"
        return context

    @staticmethod
    def _merge_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse the leading system messages into one (some NIM models accept only one)."""
        system_parts, rest, leading = [], [], True
        for m in messages:
            if leading and m["role"] == "system":
                system_parts.append(m["content"])
            else:
                leading = False
                rest.append(m)
        if not system_parts:
            return rest
        return [{"role": "system", "content": "\n\n".join(system_parts)}] + rest

    def _preflight(self, compactor: ContextCompactor) -> Dict[str, Any]:
        """Estimate tokens before a NIM call and compact until the request fits the window."""
        budget = compactor.check_token_budget(RESPONSE_TOKEN_RESERVE)
        if not budget["within_budget"]:
            compactor.compact()
            budget = compactor.check_token_budget(RESPONSE_TOKEN_RESERVE)
        if not budget["within_budget"]:
            compactor.force_compact(max(compactor.max_tokens - RESPONSE_TOKEN_RESERVE, 0))
            budget = compactor.check_token_budget(RESPONSE_TOKEN_RESERVE)
        if not budget["within_budget"]:
            compactor.truncate_large_results()
            budget = compactor.check_token_budget(RESPONSE_TOKEN_RESERVE)
        self.last_context_summary = compactor.get_context_summary()
        return budget

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
            """Run a tool in a worker thread so slow commands don't block the event loop.
        
            Integrates hooks (pre/post) and telemetry recording.
            """
            start_time = time.time()
        
            # Use hooked sandbox if available
            sandbox = self._hooked_sandbox or self.sandbox
        
            # Record tool call for telemetry
            tool_start = time.time()
        
            try:
                if self._hooked_sandbox:
                    # Hooked sandbox handles pre/post hooks internally
                    if name == "run_terminal":
                        result = await self._hooked_sandbox.run_terminal(arguments.get("command", ""), arguments.get("cwd"))
                        # Convert TerminalResult to string for compatibility
                        if isinstance(result, dict):
                            output = f"Exit code: {result.get('exit_code', 0)}\n{result.get('stdout', '')}\n{result.get('stderr', '')}"
                            if result.get('timed_out'):
                                output += f"\n[Timed out after {result.get('duration_ms', 0)}ms]"
                        else:
                            output = str(result)
                    elif name == "apply_edit":
                        output = await self._hooked_sandbox.apply_edit(
                            arguments.get("path", ""),
                            arguments.get("search_block", ""),
                            arguments.get("replace_block", "")
                        )
                    elif name == "create_file":
                        output = await self._hooked_sandbox.create_file(
                            arguments.get("path", ""),
                            arguments.get("content", "")
                        )
                    else:
                        # Delegate to original sandbox for other tools
                        output = await asyncio.to_thread(
                            lambda: asyncio.run(execute_tool_call(sandbox, name, arguments))
                        )
                else:
                    output = await asyncio.to_thread(
                        lambda: asyncio.run(execute_tool_call(sandbox, name, arguments))
                    )
            
                duration_ms = (time.time() - tool_start) * 1000
            
                # Record telemetry
                if self._telemetry_context:
                    exit_code = None
                    stdout = ""
                    stderr = ""
                    success = True
                
                    if isinstance(output, dict):
                        exit_code = output.get("exit_code")
                        stdout = output.get("stdout", "")
                        stderr = output.get("stderr", "")
                        success = exit_code == 0
                    elif "error" in str(output).lower() or "failed" in str(output).lower():
                        success = False
                
                    await self._telemetry_context.collector.record_tool_call_async(
                        tool_name=name,
                        args=arguments,
                        duration_ms=duration_ms,
                        exit_code=exit_code,
                        stdout=stdout,
                        stderr=stderr,
                        success=success,
                    )
                
                    # Check for redundant calls
                    args_hash = self._telemetry_context.detector._hash_args(arguments)
                    warning = await self._telemetry_context.detector.record_call_async(name, args_hash)
                    if warning:
                        logger.warning(f"Redundant call detected: {name} called {warning.call_count} times with same args")
            
                return output
            
            except Exception as e:
                duration_ms = (time.time() - tool_start) * 1000
                if self._telemetry_context:
                    await self._telemetry_context.collector.record_tool_call_async(
                        tool_name=name,
                        args=arguments,
                        duration_ms=duration_ms,
                        exit_code=-1,
                        stdout="",
                        stderr=str(e),
                        success=False,
                    )
                raise

    # ==================== Core model/tool loop ====================

    async def _run_phase(
        self,
        role: str,
        model: str,
        system_prompt: str,
        user_content: str,
        tools: List[Dict[str, Any]],
        repo_map: str,
        result: _PhaseResult,
        plan: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Drive one role until the model stops calling tools.

        Yields event dicts ("stream", "tool_call", "tool_result", "context", "warning");
        the final content/tool calls are stored in `result`.
        """
        ctx = self.settings_manager.settings.context
        compactor = create_context_compactor(
            max_tokens=ctx.window_tokens,
            keep_last_turns=ctx.keep_last_turns,
            summary_max_lines=ctx.summary_max_lines,
            token_method=ctx.token_estimation_method,
        )
        compactor.set_system_prompt(system_prompt)
        compactor.set_repo_map(repo_map)
        if plan:
            compactor.set_architectural_plan(plan)
        compactor.add_user_message(user_content)

        allowed = {t["function"]["name"] for t in tools}
        final_text_parts: List[str] = []
        # Track LLM call for telemetry
        llm_start_time = time.time()
        prompt_tokens_estimate = 0  # Would need tiktoken for accurate count

        for _ in range(MAX_TOOL_ROUNDS):
            for _ in range(MAX_TOOL_ROUNDS):
                budget = self._preflight(compactor)
                yield {
                    "type": "context", "step": role,
                    "tokens": budget["current_tokens"], "max_tokens": budget["max_tokens"],
                    "compressed_turns": self.last_context_summary.get("compressed_turns", 0),
                }
                if not budget["within_budget"]:
                    yield {"type": "warning", "message": (
                        f"Context still above budget ({budget['projected_tokens']}/{budget['max_tokens']} tokens) "
                        "after compaction; proceeding anyway."
                    )}

                messages = self._merge_system_messages(compactor.build_messages(include_reasoning=False))

                content, reasoning = "", ""
                pending: Dict[int, Dict[str, str]] = {}
                async for chunk in self._stream_chat_completion(model, messages, tools=tools):
                    data = self._parse_stream_chunk(chunk)
                    if "error" in data:
                        raise RuntimeError(f"NIM error: {data['error']}")
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        content += delta["content"]
                        yield {"type": "stream", "step": role, "content": delta["content"]}
                    if delta.get("reasoning_content"):
                        reasoning += delta["reasoning_content"]
                    for tc in delta.get("tool_calls") or []:
                        slot = pending.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        func = tc.get("function") or {}
                        if func.get("name"):
                            slot["name"] = func["name"]
                        if func.get("arguments"):
                            slot["arguments"] += func["arguments"]

                result.reasoning += reasoning
                if content:
                    final_text_parts = [content]

                calls = [pending[i] for i in sorted(pending) if pending[i]["name"]]
                for i, c in enumerate(calls):
                    c["id"] = c["id"] or f"call_{len(result.tool_calls) + i}"

                if not calls:
                    compactor.add_assistant_message(content, reasoning=reasoning)
                    break

                compactor.add_assistant_message(
                    content,
                    tool_calls=[
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                        for c in calls
                    ],
                    reasoning=reasoning,
                )

                for c in calls:
                    try:
                        args = json.loads(c["arguments"] or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be a JSON object")
                    except ValueError as e:
                        args = {}
                        output = f"Invalid tool arguments for {c['name']}: {e}"
                    else:
                        if c["name"] not in allowed:
                            output = f"Tool '{c['name']}' is not available in this mode."
                        else:
                            output = await self._execute_tool(c["name"], args)
                            if c["name"] in MUTATING_TOOLS:
                                self._repo_map_dirty = True
                            elif c["name"] == "run_terminal":
                                m = re.match(r"Exit code: (-?\d+)", output)
                                result.last_exit_code = int(m.group(1)) if m else result.last_exit_code

                    result.tool_calls.append(ToolCall(id=c["id"], name=c["name"], arguments=args))
                    compactor.add_tool_result(c["id"], c["name"], output)
                    yield {"type": "tool_call", "step": role, "name": c["name"], "arguments": args}
                    yield {"type": "tool_result", "step": role, "name": c["name"], "content": output}
            else:
                yield {"type": "warning", "message": f"{role}: stopped after {MAX_TOOL_ROUNDS} tool rounds."}

            # Record LLM telemetry
            if self._telemetry_context:
                llm_duration_ms = (time.time() - llm_start_time) * 1000
                # Estimate tokens from context (rough approximation)
                total_chars = sum(len(m.get("content", "")) for m in messages)
                estimated_prompt_tokens = total_chars // 4
                estimated_completion_tokens = len(content) // 4
            
                await self._telemetry_context.collector.record_llm_call_async(
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=estimated_completion_tokens,
                    reasoning_tokens=len(reasoning) // 4,
                    latency_ms=llm_duration_ms,
                    model=model,
                    role=role,
                )

            result.content = "".join(final_text_parts)
            result.messages = compactor.build_messages(include_reasoning=False)

    def _record_step(self, step_type: str, role: str, result: _PhaseResult) -> None:
        self._steps.append(AgentStep(
            step_type=step_type,
            model=self._get_model_for_role(role),
            input_messages=[
                AgentMessage(role=m["role"], content=m.get("content") or "") for m in result.messages
            ],
            output=result.content,
            tool_calls=result.tool_calls,
            reasoning=result.reasoning,
        ))

    # ==================== Public modes ====================

    async def run_ask_mode(
        self, user_prompt: str, context_files: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        """Ask/Chat mode: read-only Q&A. Yields text chunks and tool traces."""
        system_prompt = (
            "You are an expert software engineer. Answer questions about the codebase using the provided "
            "context and the <repo_map>.\nYou can use list_files and read_file to explore the workspace "
            "further if needed.\nBe concise and accurate."
        )
        repo_map = await self._build_repo_map(f"{user_prompt} {' '.join(context_files or [])}")
        result = _PhaseResult()
        user_content = f"Question: {user_prompt}\n\nContext:{self._read_context_files(context_files)}"

        async for event in self._run_phase(
            "ask", self._get_model_for_role("coder"), system_prompt, user_content,
            self._read_only_tools(), repo_map, result,
        ):
            if event["type"] == "stream":
                yield event["content"]
            elif event["type"] == "tool_result":
                yield f"\n[Tool: {event['name']}]\n{event['content']}\n"

    async def run_agent_mode(
        self,
        user_prompt: str,
        context_files: Optional[List[str]] = None,
        test_command: str = "pytest -v",
        max_iterations: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Agent/Codex mode: checkpoint -> plan -> edit -> verify -> (fix loop).

        Yields progress event dicts: step_start, stream, tool_call, tool_result, context,
        warning, step_complete, checkpoint, success, failure, done.
        """
        self._steps = []
        max_iter = max_iterations or self.max_iterations
        settings = self.settings_manager.settings
        query = f"{user_prompt} {' '.join(context_files or [])}"
        file_context = self._read_context_files(context_files)

        # Shadow checkpoint BEFORE the loop so the whole run can be undone with one click.
        self._last_checkpoint = None
        if settings.sandbox.auto_git_checkpoint and self.git_guard:
            try:
                self._last_checkpoint = await asyncio.to_thread(
                    self.git_guard.create_checkpoint, f"Pre-agent: {user_prompt[:60]}"
                )
                yield {"type": "checkpoint", "checkpoint_id": self._last_checkpoint.checkpoint_id}
            except Exception as e:
                yield {"type": "warning", "message": f"Git checkpoint failed; rollback unavailable for this run: {e}"}

        # ---- Plan ----
        yield {"type": "step_start", "step": "plan", "iteration": 0}
        repo_map = await self._build_repo_map(query)
        plan_result = _PhaseResult()
        async for ev in self._run_phase(
            "planner", self._get_model_for_role("planner"), self._build_system_prompt("planner"),
            f"Task: {user_prompt}\n\nWorkspace Context:{file_context}",
            self._read_only_tools(), repo_map, plan_result,
        ):
            yield {**ev, "step": "plan"}
        plan = plan_result.content
        self._record_step("plan", "planner", plan_result)
        yield {"type": "step_complete", "step": "plan", "content": plan}

        # ---- Edit ----
        yield {"type": "step_start", "step": "edit", "iteration": 0}
        repo_map = await self._build_repo_map(query)
        edit_result = _PhaseResult()
        async for ev in self._run_phase(
            "coder", self._get_model_for_role("coder"), self._build_system_prompt("coder"),
            f"Plan:\n{plan}\n\n{file_context}",
            TOOL_SCHEMAS, repo_map, edit_result, plan=plan,
        ):
            yield {**ev, "step": "edit"}
        self._record_step("edit", "coder", edit_result)
        yield {"type": "step_complete", "step": "edit", "content": edit_result.content}

        # ---- Verify + self-heal ----
        iteration = 0
        while iteration < max_iter:
            yield {"type": "step_start", "step": "verify", "iteration": iteration}
            repo_map = await self._build_repo_map(query)
            verify_result = _PhaseResult()
            async for ev in self._run_phase(
                "verifier", self._get_model_for_role("verifier"), self._build_system_prompt("verifier"),
                f"Run verification command: {test_command}",
                TOOL_SCHEMAS, repo_map, verify_result, plan=plan,
            ):
                yield {**ev, "step": "verify"}
            self._record_step("verify", "verifier", verify_result)
            verify_output = verify_result.content
            yield {"type": "step_complete", "step": "verify", "content": verify_output}

            if self._verification_passed(verify_result):
                yield {"type": "success", "message": "All tests passed!", "iterations": iteration + 1}
                break

            if iteration >= max_iter - 1:
                yield {"type": "failure",
                       "message": f"Max iterations ({max_iter}) reached. Tests still failing.",
                       "iterations": iteration + 1}
                break

            yield {"type": "step_start", "step": "fix", "iteration": iteration}
            repo_map = await self._build_repo_map(query)
            fix_result = _PhaseResult()
            fix_prompt = (
                f"Previous verification failed with errors:\n{verify_output}\n\n"
                "Please fix the issues. The verifier will re-run afterwards.\n\n" + file_context
            )
            async for ev in self._run_phase(
                "coder", self._get_model_for_role("coder"),
                self._build_system_prompt("coder") + "\n\nYou are now in FIX mode. Analyze the errors and apply corrections.",
                fix_prompt, TOOL_SCHEMAS, repo_map, fix_result, plan=plan,
            ):
                yield {**ev, "step": "fix"}
            self._record_step("fix", "coder", fix_result)
            yield {"type": "step_complete", "step": "fix", "content": fix_result.content}
            iteration += 1

        yield {
            "type": "done",
            "steps": self._steps,
            "checkpoint_id": self._last_checkpoint.checkpoint_id if self._last_checkpoint else None,
        }

    @staticmethod
    def _verification_passed(result: _PhaseResult) -> bool:
        """Prefer the real exit code of the last terminal command; fall back to the verifier's text."""
        if result.last_exit_code is not None:
            return result.last_exit_code == 0
        text = result.content.lower()
        return "passed" in text and "fail" not in text and "error" not in text

    # ==================== New Integration Accessors ====================

    def get_telemetry_data(self) -> Optional[ProfilerData]:
                """Get aggregated telemetry data for UI visualization."""
                if self._telemetry_context:
                    return ProfilerData.from_context(self._telemetry_context)
                return None

    def get_mcp_status(self) -> Dict[str, Any]:
                """Get MCP connection status for UI."""
                if self._mcp_manager:
                    return self._mcp_manager.get_connection_status()
                return {}

    def get_hook_status(self) -> Dict[str, Any]:
                """Get hook system status for UI."""
                if self._hook_registry:
                    return {
                        "enabled": True,
                        "pre_hooks": list(self._hook_registry._pre_hooks.keys()),
                        "post_hooks": list(self._hook_registry._post_hooks.keys()),
                    }
                return {"enabled": False}

    def get_sub_agent_status(self) -> Dict[str, Any]:
                """Get sub-agent status for UI."""
                if self._sub_agent_manager:
                    return {
                        "explorer": "ready",
                        "coder": "ready",
                        "test_writer": "ready",
                    }
                return {}

    # ==================== State accessors ====================

    def get_conversation_history(self) -> List[AgentMessage]:
                return self._conversation

    def get_steps(self) -> List[AgentStep]:
                return self._steps

    def clear_history(self) -> None:
                self._conversation = []
                self._steps = []


def build_sandbox(settings_manager: Optional[SettingsManager] = None, **overrides: Any) -> SandboxManager:
            """Create a SandboxManager from persisted settings (optionally overridden)."""
            sm = settings_manager or get_settings_manager()
            s = sm.settings.sandbox
            return SandboxManager(
                workspace_dir=overrides.get("workspace_dir", s.workspace_dir),
                sandbox_type=SandboxType(overrides.get("sandbox_type", s.sandbox_type)),
                docker_image=overrides.get("docker_image", s.docker_image),
                docker_cpu_limit=s.docker_cpu_limit,
                docker_memory_limit=s.docker_memory_limit,
                command_timeout=overrides.get("command_timeout", s.command_timeout),
                auto_approve=overrides.get("auto_approve", s.auto_approve_commands),
    )


async def create_harness(
            workspace_dir: str,
            mode: AgentMode = AgentMode.ASK,
            max_iterations: int = 5,
            auto_approve: bool = False,
            command_timeout: int = 120,
        ) -> AgentHarness:
            """Factory: build and enter an AgentHarness (callers must await its __aexit__)."""
            settings_manager = get_settings_manager()
            sandbox = build_sandbox(
                settings_manager, workspace_dir=workspace_dir,
                auto_approve=auto_approve, command_timeout=command_timeout,
    )
            git_guard = GitGuard(workspace_dir) if settings_manager.settings.sandbox.auto_git_checkpoint else None
            harness = AgentHarness(sandbox, settings_manager, mode, max_iterations, git_guard)
            await harness.__aenter__()
            return harness
