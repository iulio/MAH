"""
Context-Isolated Specialist Sub-Agents for NIM Agentic Coding Assistant.

Implements a multi-agent architecture with isolated conversation contexts:
- BaseSubAgent: Abstract base class with isolated history, context compactor, repo map access
- ExplorerSubAgent: Read-only codebase exploration and architectural questions
- CoderSubAgent: Full tool access for implementing module/file changes
- TestWriterSubAgent: Writes/updates unit tests in parallel with Coder
- SubAgentManager: Coordinates sub-agents with ThreadPoolExecutor for parallel execution
- Integration with AgentHarness: Planner → Explorer → Coder + TestWriter (parallel) → Verifier
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from pathlib import Path

from core.config import SettingsManager, get_settings_manager
from core.context_manager import ContextCompactor, create_context_compactor, MessageRole
from core.repo_map import RepoMapper, create_repo_mapper
from sandbox.executor import SandboxManager, TOOL_SCHEMAS, execute_tool_call

logger = logging.getLogger(__name__)

# Tool access control
READ_ONLY_TOOLS = ("list_files", "read_file", "run_terminal")
MUTATING_TOOLS = ("apply_edit", "create_file")
ALL_TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}
READ_ONLY_TOOL_SET = set(READ_ONLY_TOOLS)
MUTATING_TOOL_SET = set(MUTATING_TOOLS)


class SubAgentRole(str, Enum):
    """Sub-agent role types."""
    EXPLORER = "explorer"
    CODER = "coder"
    TEST_WRITER = "test_writer"


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""
    success: bool
    summary: str  # Distilled output for Supervisor
    files_modified: List[str] = field(default_factory=list)
    diffs: List[Dict[str, Any]] = field(default_factory=list)  # file, diff, status
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""
    role: SubAgentRole
    model: str
    system_prompt: str
    allowed_tools: Set[str]
    max_tool_rounds: int = 15
    timeout_seconds: int = 300
    context_window_tokens: int = 8000
    keep_last_turns: int = 3
    summary_max_lines: int = 2


class BaseSubAgent(ABC):
    """
    Abstract base class for all sub-agents.
    
    Provides:
    - Isolated conversation history (own message list)
    - Own context compactor and repo map access
    - Model assignment per role
    - Async run(task: str, context: Dict) -> SubAgentResult
    - Tool access control (read-only vs full)
    """
    
    def __init__(
        self,
        config: SubAgentConfig,
        sandbox: SandboxManager,
        settings_manager: SettingsManager,
        repo_mapper: Optional[RepoMapper] = None,
    ):
        self.config = config
        self.sandbox = sandbox
        self.settings_manager = settings_manager
        self.repo_mapper = repo_mapper
        self._nim_client = None
        self._conversation: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._files_modified: Set[str] = set()
        self._diffs: List[Dict[str, Any]] = []
        self._reasoning: str = ""
        self._agent_id = str(uuid.uuid4())[:8]
        
        # Initialize context compactor
        self.compactor = create_context_compactor(
            max_tokens=config.context_window_tokens,
            keep_last_turns=config.keep_last_turns,
            summary_max_lines=config.summary_max_lines,
            token_method=settings_manager.settings.context.token_estimation_method,
        )
        self.compactor.set_system_prompt(config.system_prompt)
    
    async def __aenter__(self) -> "BaseSubAgent":
        from core.config import NIMClient
        self._nim_client = NIMClient(self.settings_manager.settings.nim)
        await self._nim_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._nim_client:
            await self._nim_client.__aexit__(exc_type, exc_val, exc_tb)
            self._nim_client = None
    
    @property
    def agent_id(self) -> str:
        return self._agent_id
    
    @property
    def role(self) -> SubAgentRole:
        return self.config.role
    
    @abstractmethod
    def _build_user_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Build the user prompt for this sub-agent's task."""
        pass
    
    @abstractmethod
    def _get_repo_map_query(self, task: str, context: Dict[str, Any]) -> str:
        """Get the query for repo map relevance ranking."""
        pass
    
    def _get_allowed_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas filtered by allowed tools."""
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in self.config.allowed_tools]
    
    def _merge_system_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse leading system messages into one."""
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
    
    async def _stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        if not self._nim_client:
            raise RuntimeError("NIM client not initialized; use 'async with SubAgent(...)'")
        async for chunk in self._nim_client.chat_completion_stream(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=temperature,
        ):
            yield chunk
    
    @staticmethod
    def _parse_stream_chunk(chunk: str) -> Dict[str, Any]:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return {}
    
    def _preflight(self, estimated_response_tokens: int = 2000) -> Dict[str, Any]:
        """Pre-flight token budget check."""
        budget = self.compactor.check_token_budget(estimated_response_tokens)
        if not budget["within_budget"]:
            self.compactor.compact()
            budget = self.compactor.check_token_budget(estimated_response_tokens)
        if not budget["within_budget"]:
            self.compactor.force_compact(max(self.compactor.max_tokens - estimated_response_tokens, 0))
            budget = self.compactor.check_token_budget(estimated_response_tokens)
        if not budget["within_budget"]:
            self.compactor.truncate_large_results()
            budget = self.compactor.check_token_budget(estimated_response_tokens)
        return budget
    
    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call."""
        return await asyncio.to_thread(
            lambda: asyncio.run(execute_tool_call(self.sandbox, name, arguments))
        )
    
    async def _run_tool_loop(
        self,
        user_prompt: str,
        repo_map: str,
    ) -> SubAgentResult:
        """Run the model/tool loop until completion."""
        self.compactor.set_repo_map(repo_map)
        self.compactor.add_user_message(user_prompt)
        
        allowed_tools = self.config.allowed_tools
        tool_schemas = self._get_allowed_tool_schemas()
        
        final_content = ""
        
        for round_num in range(self.config.max_tool_rounds):
            budget = self._preflight()
            if not budget["within_budget"]:
                logger.warning(
                    f"[{self.role.value}] Context over budget after compaction: "
                    f"{budget['projected_tokens']}/{budget['max_tokens']}"
                )
            
            messages = self._merge_system_messages(
                self.compactor.build_messages(include_reasoning=False)
            )
            
            content, reasoning = "", ""
            pending: Dict[int, Dict[str, str]] = {}
            
            async for chunk in self._stream_chat_completion(messages, tools=tool_schemas):
                data = self._parse_stream_chunk(chunk)
                if "error" in data:
                    raise RuntimeError(f"NIM error: {data['error']}")
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content += delta["content"]
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
            
            self._reasoning += reasoning
            if content:
                final_content = content
            
            calls = [pending[i] for i in sorted(pending) if pending[i]["name"]]
            for i, c in enumerate(calls):
                c["id"] = c["id"] or f"call_{len(self._tool_calls) + i}"
            
            if not calls:
                self.compactor.add_assistant_message(content, reasoning=reasoning)
                break
            
            # Record tool calls
            tool_call_records = []
            for c in calls:
                try:
                    args = json.loads(c["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as e:
                    args = {}
                    output = f"Invalid tool arguments for {c['name']}: {e}"
                else:
                    if c["name"] not in allowed_tools:
                        output = f"Tool '{c['name']}' is not available for {self.role.value}."
                    else:
                        output = await self._execute_tool(c["name"], args)
                        # Track file modifications
                        if c["name"] in MUTATING_TOOL_SET:
                            self._files_modified.add(args.get("path", "unknown"))
                            if c["name"] == "apply_edit":
                                self._diffs.append({
                                    "file": args.get("path", "unknown"),
                                    "diff": output,
                                    "status": "success" if not output.startswith("Tool execution error") else "error"
                                })
                
                tool_call_records.append({
                    "id": c["id"],
                    "name": c["name"],
                    "arguments": args,
                    "result": output
                })
                self._tool_calls.append({
                    "id": c["id"],
                    "name": c["name"],
                    "arguments": args,
                    "result": output
                })
            
            # Add assistant message with tool calls
            self.compactor.add_assistant_message(
                content,
                tool_calls=[
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in calls
                ],
                reasoning=reasoning,
            )
            
            # Add tool results
            for tc in tool_call_records:
                self.compactor.add_tool_result(
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    result=tc["result"]
                )
        
        return SubAgentResult(
            success=True,
            summary=final_content,
            files_modified=list(self._files_modified),
            diffs=self._diffs,
            tool_calls=self._tool_calls,
            reasoning=self._reasoning,
        )
    
    async def run(self, task: str, context: Dict[str, Any]) -> SubAgentResult:
        """
        Execute the sub-agent's task.
        
        Args:
            task: The task description
            context: Additional context (plan, file_context, etc.)
            
        Returns:
            SubAgentResult with execution outcome
        """
        try:
            # Build repo map for this task
            query = self._get_repo_map_query(task, context)
            repo_map = ""
            if self.repo_mapper:
                repo_map = await asyncio.to_thread(self.repo_mapper.format_repo_map, query=query)
            
            # Build user prompt
            user_prompt = self._build_user_prompt(task, context)
            
            # Run the tool loop
            result = await self._run_tool_loop(user_prompt, repo_map)
            
            return result
            
        except Exception as e:
            logger.exception(f"[{self.role.value}] Sub-agent execution failed")
            return SubAgentResult(
                success=False,
                summary=f"Sub-agent {self.role.value} failed: {str(e)}",
                error=str(e),
                reasoning=self._reasoning,
                tool_calls=self._tool_calls,
            )


class ExplorerSubAgent(BaseSubAgent):
    """
    Read-only sub-agent for codebase exploration and architectural questions.
    
    Tools: list_files, read_file, run_terminal (read-only commands like cat, grep, find)
    Model: verifier_model (fast, analytical)
    Returns: Distilled summary to Supervisor
    """
    
    def __init__(
        self,
        sandbox: SandboxManager,
        settings_manager: SettingsManager,
        repo_mapper: Optional[RepoMapper] = None,
    ):
        models = settings_manager.settings.models
        config = SubAgentConfig(
            role=SubAgentRole.EXPLORER,
            model=models.verifier_model,
            system_prompt="""You are an expert codebase explorer and architectural analyst.
Your job is to answer questions about the codebase structure, find relevant files,
and provide distilled architectural summaries.

Guidelines:
- Use list_files to explore directory structure
- Use read_file to examine specific files
- Use run_terminal for READ-ONLY commands: grep, find, cat, head, tail, wc, tree
- NEVER use apply_edit, create_file, or any mutating commands
- Synthesize findings into a clear, concise summary for the Supervisor
- Focus on: file locations, key classes/functions, dependencies, patterns, and architecture
- Be precise and cite file paths""",
            allowed_tools=READ_ONLY_TOOL_SET,
            max_tool_rounds=10,
            timeout_seconds=180,
        )
        super().__init__(config, sandbox, settings_manager, repo_mapper)
    
    def _build_user_prompt(self, task: str, context: Dict[str, Any]) -> str:
        plan = context.get("plan", "")
        file_context = context.get("file_context", "")
        return (
                    f"Task: {task}\n\n"
                    f"{f'Planner\\'s Plan:\\n{plan}\\n' if plan else ''}"
                    f"{f'File Context:\\n{file_context}\\n' if file_context else ''}"
                    "Explore the codebase to gather context for this task. Focus on:\n"
                    "1. Finding all relevant files and their locations\n"
                    "2. Understanding the architectural patterns used\n"
                    "3. Identifying key classes, functions, and dependencies\n"
                    "4. Noting any existing tests or similar implementations\n\n"
                    "Return a distilled summary with file paths and key findings."
                )
    
    def _get_repo_map_query(self, task: str, context: Dict[str, Any]) -> str:
        plan = context.get("plan", "")
        return f"{task} {plan}".strip()


class CoderSubAgent(BaseSubAgent):
    """
    Full-access sub-agent for implementing module/file changes.
    
    Tools: apply_edit, create_file, run_terminal, list_files, read_file
    Model: coder_model (code generation optimized)
    Returns: Diffs and file paths modified
    """
    
    def __init__(
        self,
        sandbox: SandboxManager,
        settings_manager: SettingsManager,
        repo_mapper: Optional[RepoMapper] = None,
    ):
        models = settings_manager.settings.models
        config = SubAgentConfig(
            role=SubAgentRole.CODER,
            model=models.coder_model,
            system_prompt="""You are an expert software engineer implementing code changes.
Your job is to execute the plan by making precise file edits.

Guidelines:
- Use apply_edit for surgical changes (provide exact search_block and replace_block)
- Use create_file for new files
- Python edits are syntax-checked before saving; if an edit is rejected, fix it and retry
- Make minimal, focused changes following the plan exactly
- Follow the existing code style and patterns
- If you need to see a file first, use read_file
- Use run_terminal to run tests or verify changes
- Always verify your changes make sense in context
- Output tool calls to execute the changes""",
            allowed_tools=ALL_TOOL_NAMES,
            max_tool_rounds=20,
            timeout_seconds=300,
        )
        super().__init__(config, sandbox, settings_manager, repo_mapper)
    
    def _build_user_prompt(self, task: str, context: Dict[str, Any]) -> str:
        plan = context.get("plan", "")
        file_context = context.get("file_context", "")
        explorer_summary = context.get("explorer_summary", "")
        return f"""Task: {task}

Plan to implement:
{plan}

{f"Explorer Summary:\n{explorer_summary}\n" if explorer_summary else ""}
{f"File Context:\n{file_context}\n" if file_context else ""}

Implement the plan by making the necessary file changes. Follow the plan step by step.
Return a summary of changes made with file paths and diffs."""
    
    def _get_repo_map_query(self, task: str, context: Dict[str, Any]) -> str:
        plan = context.get("plan", "")
        return f"{task} {plan}".strip()


class TestWriterSubAgent(BaseSubAgent):
    """
    Sub-agent for writing/updating unit tests based on Planner's spec.
    
    Tools: apply_edit, create_file, run_terminal, list_files, read_file
    Targets: test files only (test_*.py, *_test.py, tests/)
    Model: coder_model (code generation optimized)
    Runs in parallel with CoderSubAgent via ThreadPoolExecutor
    Returns: Test file paths created/modified
    """
    
    def __init__(
        self,
        sandbox: SandboxManager,
        settings_manager: SettingsManager,
        repo_mapper: Optional[RepoMapper] = None,
    ):
        models = settings_manager.settings.models
        config = SubAgentConfig(
            role=SubAgentRole.TEST_WRITER,
            model=models.coder_model,
            system_prompt="""You are an expert test engineer writing unit tests.
Your job is to create or update test files based on the implementation plan.

Guidelines:
- Target test files ONLY: test_*.py, *_test.py, files in tests/ directories
- Use pytest conventions (fixtures, parametrize, descriptive test names)
- Write tests that verify the behavior specified in the plan
- Follow existing test patterns in the codebase
- Use apply_edit to add tests to existing test files
- Use create_file for new test files
- Run tests with pytest to verify they pass
- Do NOT modify production code (non-test files)
- Focus on: happy paths, edge cases, error conditions, boundary values""",
            allowed_tools=ALL_TOOL_NAMES,
            max_tool_rounds=15,
            timeout_seconds=240,
        )
        super().__init__(config, sandbox, settings_manager, repo_mapper)
    
    def _build_user_prompt(self, task: str, context: Dict[str, Any]) -> str:
        plan = context.get("plan", "")
        file_context = context.get("file_context", "")
        coder_changes = context.get("coder_changes", "")
        return f"""Task: {task}

Plan (for test specification):
{plan}

{f"Coder's Changes:\n{coder_changes}\n" if coder_changes else ""}
{f"File Context:\n{file_context}\n" if file_context else ""}

Write or update unit tests for the implementation. Target test files only:
- test_*.py
- *_test.py  
- tests/**/*.py

Follow the existing test patterns. Run pytest to verify tests pass.
Return a summary of test files created/modified."""
    
    def _get_repo_map_query(self, task: str, context: Dict[str, Any]) -> str:
        plan = context.get("plan", "")
        return f"tests {task} {plan}".strip()
    
    def _is_test_file(self, path: str) -> bool:
        """Check if a file path is a test file."""
        path_lower = path.lower()
        return (
            path_lower.startswith("test_") or
            path_lower.endswith("_test.py") or
            "tests/" in path_lower or
            "/test_" in path_lower
        )


class SubAgentManager:
    """
    Coordinates sub-agents for the multi-agent workflow.
    
    - Spawns sub-agents for each phase
    - ThreadPoolExecutor for parallel Coder + TestWriter execution
    - Results aggregated back to main conversation
    - Handles timeouts and error propagation
    """
    
    def __init__(
        self,
        sandbox: SandboxManager,
        settings_manager: SettingsManager,
        repo_mapper: Optional[RepoMapper] = None,
        max_workers: int = 2,
    ):
        self.sandbox = sandbox
        self.settings_manager = settings_manager
        self.repo_mapper = repo_mapper
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._sub_agents: Dict[SubAgentRole, BaseSubAgent] = {}
    
    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
    
    def _create_explorer(self) -> ExplorerSubAgent:
        return ExplorerSubAgent(self.sandbox, self.settings_manager, self.repo_mapper)
    
    def _create_coder(self) -> CoderSubAgent:
        return CoderSubAgent(self.sandbox, self.settings_manager, self.repo_mapper)
    
    def _create_test_writer(self) -> TestWriterSubAgent:
        return TestWriterSubAgent(self.sandbox, self.settings_manager, self.repo_mapper)
    
    async def run_explorer(self, task: str, context: Dict[str, Any]) -> SubAgentResult:
        """Run the Explorer sub-agent for context gathering."""
        async with self._create_explorer() as agent:
            return await agent.run(task, context)
    
    async def run_coder(self, task: str, context: Dict[str, Any]) -> SubAgentResult:
        """Run the Coder sub-agent for implementation."""
        async with self._create_coder() as agent:
            return await agent.run(task, context)
    
    async def run_test_writer(self, task: str, context: Dict[str, Any]) -> SubAgentResult:
        """Run the TestWriter sub-agent for test creation."""
        async with self._create_test_writer() as agent:
            return await agent.run(task, context)
    
    async def run_coder_and_test_writer_parallel(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> tuple[SubAgentResult, SubAgentResult]:
        """
        Run Coder and TestWriter in parallel.
        
        Returns:
            Tuple of (coder_result, test_writer_result)
        """
        # Create agents
        coder = self._create_coder()
        test_writer = self._create_test_writer()
        
        async def run_coder():
            async with coder:
                return await coder.run(task, context)
        
        async def run_test_writer():
            async with test_writer:
                return await test_writer.run(task, context)
        
        # Run in parallel using asyncio.gather
        coder_result, test_writer_result = await asyncio.gather(
            run_coder(),
            run_test_writer(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(coder_result, Exception):
            logger.exception("Coder sub-agent failed")
            coder_result = SubAgentResult(
                success=False,
                summary=f"Coder failed: {coder_result}",
                error=str(coder_result)
            )
        
        if isinstance(test_writer_result, Exception):
            logger.exception("TestWriter sub-agent failed")
            test_writer_result = SubAgentResult(
                success=False,
                summary=f"TestWriter failed: {test_writer_result}",
                error=str(test_writer_result)
            )
        
        return coder_result, test_writer_result
    
    async def run_full_workflow(
        self,
        user_prompt: str,
        context_files: Optional[List[str]] = None,
        test_command: str = "pytest -v",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the full multi-agent workflow:
        Planner → Explorer → Coder + TestWriter (parallel) → Verifier
        
        Yields progress events for UI integration.
        """
        file_context = self._read_context_files(context_files or [])
        query = f"{user_prompt} {' '.join(context_files or [])}"
        
        # ---- Phase 1: Planner (using existing AgentHarness planner) ----
        yield {"type": "step_start", "step": "plan", "phase": "planner"}
        # Note: Planner is run via AgentHarness.run_agent_mode's plan phase
        # This manager handles post-plan phases
        
        # For now, we expect the plan to be passed in context
        plan = context.get("plan", "") if (context := {}) else ""
        
        # ---- Phase 2: Explorer (context gathering) ----
        yield {"type": "step_start", "step": "explore", "phase": "explorer"}
        explorer_context = {
            "plan": plan,
            "file_context": file_context,
        }
        explorer_result = await self.run_explorer(user_prompt, explorer_context)
        
        yield {
            "type": "step_complete",
            "step": "explore",
            "phase": "explorer",
            "success": explorer_result.success,
            "summary": explorer_result.summary,
            "files_examined": explorer_result.files_modified,  # Actually files read
        }
        
        if not explorer_result.success:
            yield {"type": "failure", "step": "explore", "error": explorer_result.error}
            return
        
        # ---- Phase 3: Coder + TestWriter (parallel) ----
        yield {"type": "step_start", "step": "implement", "phase": "coder+test_writer"}
        
        implementation_context = {
            "plan": plan,
            "file_context": file_context,
            "explorer_summary": explorer_result.summary,
        }
        
        coder_result, test_writer_result = await self.run_coder_and_test_writer_parallel(
            user_prompt, implementation_context
        )
        
        yield {
            "type": "step_complete",
            "step": "implement",
            "phase": "coder",
            "success": coder_result.success,
            "summary": coder_result.summary,
            "files_modified": coder_result.files_modified,
            "diffs": coder_result.diffs,
        }
        
        yield {
            "type": "step_complete",
            "step": "implement",
            "phase": "test_writer",
            "success": test_writer_result.success,
            "summary": test_writer_result.summary,
            "files_modified": test_writer_result.files_modified,
            "diffs": test_writer_result.diffs,
        }
        
        if not coder_result.success:
            yield {"type": "failure", "step": "coder", "error": coder_result.error}
            return
        
        # ---- Phase 4: Verifier ----
        yield {"type": "step_start", "step": "verify", "phase": "verifier"}
        
        # Run verification command
        verify_result = await self._run_verification(test_command)
        
        yield {
            "type": "step_complete",
            "step": "verify",
            "phase": "verifier",
            "success": verify_result.success,
            "summary": verify_result.summary,
        }
        
        if verify_result.success:
            yield {"type": "success", "message": "All phases completed successfully!"}
        else:
            yield {"type": "failure", "step": "verify", "error": verify_result.error}
        
        yield {
            "type": "done",
            "explorer": explorer_result,
            "coder": coder_result,
            "test_writer": test_writer_result,
            "verifier": verify_result,
        }
    
    async def _run_verification(self, test_command: str) -> SubAgentResult:
        """Run verification command (pytest, etc.)."""
        try:
            result = await asyncio.to_thread(
                lambda: asyncio.run(execute_tool_call(self.sandbox, "run_terminal", {"command": test_command}))
            )
            
            # Parse result
            exit_code = 0
            for line in result.splitlines():
                if line.startswith("Exit code:"):
                    try:
                        exit_code = int(line.split(":")[1].strip())
                    except Exception:
                        pass
            
            success = exit_code == 0
            return SubAgentResult(
                success=success,
                summary=result,
                metadata={"exit_code": exit_code, "command": test_command}
            )
        except Exception as e:
            return SubAgentResult(
                success=False,
                summary=f"Verification failed: {e}",
                error=str(e)
            )
    
    def _read_context_files(self, context_files: List[str]) -> str:
        """Read context files for the prompt."""
        context = ""
        for f in context_files:
            try:
                content = self.sandbox.read_file(f)
                if len(content) > 20000:
                    content = content[:20000] + "\n... [truncated]"
                context += f"\n--- File: {f} ---\n{content}\n"
            except Exception as e:
                context += f"\n--- File: {f} (ERROR: {e}) ---\n"
        return context
    
    def shutdown(self):
        """Shutdown the thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


# ==================== Integration with AgentHarness ====================

class SubAgentHarness:
    """
    Extended AgentHarness that uses sub-agents for each phase.
    
    Replaces single-loop phases with sub-agent delegation:
    Planner → Explorer (context gathering) → Coder + TestWriter (parallel) → Verifier
    Each sub-agent has isolated context window.
    """
    
    def __init__(
        self,
        sandbox: SandboxManager,
        settings_manager: SettingsManager,
        max_iterations: int = 5,
        git_guard=None,
    ):
        self.sandbox = sandbox
        self.settings_manager = settings_manager
        self.max_iterations = max_iterations
        self.git_guard = git_guard
        
        # Create repo mapper
        cfg = settings_manager.settings.repo_map
        self.repo_mapper = create_repo_mapper(
            workspace_dir=str(sandbox.workspace_dir),
            token_budget=cfg.token_budget,
            include_patterns=cfg.include_patterns,
            exclude_patterns=cfg.exclude_patterns,
            max_file_size_kb=cfg.max_file_size_kb,
        )
        
        # Create sub-agent manager
        self.sub_agent_manager = SubAgentManager(
            sandbox=sandbox,
            settings_manager=settings_manager,
            repo_mapper=self.repo_mapper,
            max_workers=2,
        )
        
        # For planner/verifier (using existing patterns)
        self._nim_client = None
        self._steps = []
    
    async def __aenter__(self) -> "SubAgentHarness":
        from core.config import NIMClient
        self._nim_client = NIMClient(self.settings_manager.settings.nim)
        await self._nim_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._nim_client:
            await self._nim_client.__aexit__(exc_type, exc_val, exc_tb)
        self.sub_agent_manager.shutdown()
    
    def _get_model_for_role(self, role: str) -> str:
        models = self.settings_manager.settings.models
        return {
            "planner": models.planner_model,
            "coder": models.coder_model,
            "verifier": models.verifier_model,
        }.get(role, models.coder_model)
    
    def _build_system_prompt(self, role: str) -> str:
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
    
    async def _run_planner_phase(
        self,
        user_prompt: str,
        file_context: str,
        repo_map: str,
    ) -> str:
        """Run the planner phase using the existing pattern."""
        from core.context_manager import create_context_compactor
        from core.config import NIMClient
        
        ctx = self.settings_manager.settings.context
        compactor = create_context_compactor(
            max_tokens=ctx.window_tokens,
            keep_last_turns=ctx.keep_last_turns,
            summary_max_lines=ctx.summary_max_lines,
            token_method=ctx.token_estimation_method,
        )
        compactor.set_system_prompt(self._build_system_prompt("planner"))
        compactor.set_repo_map(repo_map)
        compactor.add_user_message(f"Task: {user_prompt}\n\nWorkspace Context:{file_context}")
        
        tools = [t for t in TOOL_SCHEMAS if t["function"]["name"] in READ_ONLY_TOOLS]
        model = self._get_model_for_role("planner")
        
        final_content = ""
        for _ in range(10):  # Max tool rounds for planner
            budget = compactor.check_token_budget(2000)
            if not budget["within_budget"]:
                compactor.compact()
            
            messages = self._merge_system_messages(compactor.build_messages(include_reasoning=False))
            
            content, reasoning = "", ""
            pending: Dict[int, Dict[str, str]] = {}
            
            async for chunk in self._nim_client.chat_completion_stream(
                model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.1
            ):
                data = self._parse_stream_chunk(chunk)
                if "error" in data:
                    raise RuntimeError(f"NIM error: {data['error']}")
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content += delta["content"]
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
            
            calls = [pending[i] for i in sorted(pending) if pending[i]["name"]]
            for i, c in enumerate(calls):
                c["id"] = c["id"] or f"call_{i}"
            
            if not calls:
                compactor.add_assistant_message(content, reasoning=reasoning)
                final_content = content
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
                except ValueError:
                    args = {}
                    output = f"Invalid tool arguments for {c['name']}"
                else:
                    if c["name"] not in READ_ONLY_TOOLS:
                        output = f"Tool '{c['name']}' not available in planner mode."
                    else:
                        output = await asyncio.to_thread(
                            lambda: asyncio.run(execute_tool_call(self.sandbox, c["name"], args))
                        )
                
                compactor.add_tool_result(c["id"], c["name"], output)
        
        return final_content
    
    @staticmethod
    def _parse_stream_chunk(chunk: str) -> Dict[str, Any]:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return {}
    
    @staticmethod
    def _merge_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    
    async def run_agent_mode_with_subagents(
        self,
        user_prompt: str,
        context_files: Optional[List[str]] = None,
        test_command: str = "pytest -v",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the full agent mode with sub-agent delegation.
        
        Flow:
        1. Git checkpoint
        2. Planner (read-only, creates plan)
        3. Explorer (read-only, gathers context)
        4. Coder + TestWriter (parallel, full access)
        5. Verifier (runs tests)
        6. Self-healing loop if verification fails
        """
        self._steps = []
        context_files = context_files or []
        file_context = self._read_context_files(context_files)
        query = f"{user_prompt} {' '.join(context_files)}"
        
        # Git checkpoint
        checkpoint_info = None
        if self.settings_manager.settings.sandbox.auto_git_checkpoint and self.git_guard:
            try:
                checkpoint_info = await asyncio.to_thread(
                    self.git_guard.create_checkpoint, f"Pre-agent: {user_prompt[:60]}"
                )
                yield {"type": "checkpoint", "checkpoint_id": checkpoint_info.checkpoint_id}
            except Exception as e:
                yield {"type": "warning", "message": f"Git checkpoint failed: {e}"}
        
        # ---- Phase 1: Planner ----
        yield {"type": "step_start", "step": "plan", "phase": "planner"}
        repo_map = await asyncio.to_thread(self.repo_mapper.format_repo_map, query=query)
        plan = await self._run_planner_phase(user_prompt, file_context, repo_map)
        
        yield {"type": "step_complete", "step": "plan", "phase": "planner", "content": plan}
        
        # ---- Phase 2: Explorer ----
        yield {"type": "step_start", "step": "explore", "phase": "explorer"}
        explorer_result = await self.sub_agent_manager.run_explorer(user_prompt, {
            "plan": plan,
            "file_context": file_context,
        })
        
        yield {
            "type": "step_complete",
            "step": "explore",
            "phase": "explorer",
            "success": explorer_result.success,
            "summary": explorer_result.summary,
        }
        
        if not explorer_result.success:
            yield {"type": "failure", "message": f"Explorer failed: {explorer_result.error}"}
            return
        
        # ---- Phase 3: Coder + TestWriter (parallel) ----
        yield {"type": "step_start", "step": "implement", "phase": "coder+test_writer"}
        
        coder_result, test_writer_result = await self.sub_agent_manager.run_coder_and_test_writer_parallel(
            user_prompt, {
                "plan": plan,
                "file_context": file_context,
                "explorer_summary": explorer_result.summary,
            }
        )
        
        yield {
            "type": "step_complete",
            "step": "implement",
            "phase": "coder",
            "success": coder_result.success,
            "summary": coder_result.summary,
            "files_modified": coder_result.files_modified,
            "diffs": coder_result.diffs,
        }
        
        yield {
            "type": "step_complete",
            "step": "implement",
            "phase": "test_writer",
            "success": test_writer_result.success,
            "summary": test_writer_result.summary,
            "files_modified": test_writer_result.files_modified,
            "diffs": test_writer_result.diffs,
        }
        
        if not coder_result.success:
            yield {"type": "failure", "message": f"Coder failed: {coder_result.error}"}
            return
        
        # ---- Phase 4: Verifier + Self-healing loop ----
        iteration = 0
        while iteration < self.max_iterations:
            yield {"type": "step_start", "step": "verify", "phase": "verifier", "iteration": iteration}
            
            repo_map = await asyncio.to_thread(self.repo_mapper.format_repo_map, query=query)
            verify_result = await self._run_verifier_phase(test_command, repo_map, plan, file_context)
            
            yield {
                "type": "step_complete",
                "step": "verify",
                "phase": "verifier",
                "iteration": iteration,
                "success": verify_result.success,
                "summary": verify_result.summary,
            }
            
            if verify_result.success:
                yield {"type": "success", "message": "All tests passed!", "iterations": iteration + 1}
                break
            
            if iteration >= self.max_iterations - 1:
                yield {"type": "failure", "message": f"Max iterations ({self.max_iterations}) reached. Tests still failing."}
                break
            
            # Fix phase - use Coder sub-agent with error context
            yield {"type": "step_start", "step": "fix", "phase": "coder", "iteration": iteration}
            
            fix_context = {
                "plan": plan,
                "file_context": file_context,
                "explorer_summary": explorer_result.summary,
                "verification_errors": verify_result.summary,
                "coder_changes": coder_result.summary,
            }
            
            fix_result = await self.sub_agent_manager.run_coder(
                f"Fix the following verification errors:\n{verify_result.summary}",
                fix_context
            )
            
            yield {
                "type": "step_complete",
                "step": "fix",
                "phase": "coder",
                "iteration": iteration,
                "success": fix_result.success,
                "summary": fix_result.summary,
                "files_modified": fix_result.files_modified,
                "diffs": fix_result.diffs,
            }
            
            if not fix_result.success:
                yield {"type": "failure", "message": f"Fix failed: {fix_result.error}"}
                break
            
            iteration += 1
        
        yield {
            "type": "done",
            "checkpoint_id": checkpoint_info.checkpoint_id if checkpoint_info else None,
            "plan": plan,
            "explorer": explorer_result,
            "coder": coder_result,
            "test_writer": test_writer_result,
            "verifier": verify_result if 'verify_result' in locals() else None,
        }
    
    async def _run_verifier_phase(
        self,
        test_command: str,
        repo_map: str,
        plan: str,
        file_context: str,
    ) -> SubAgentResult:
        """Run the verifier phase using the existing pattern."""
        from core.context_manager import create_context_compactor
        
        ctx = self.settings_manager.settings.context
        compactor = create_context_compactor(
            max_tokens=ctx.window_tokens,
            keep_last_turns=ctx.keep_last_turns,
            summary_max_lines=ctx.summary_max_lines,
            token_method=ctx.token_estimation_method,
        )
        compactor.set_system_prompt(self._build_system_prompt("verifier"))
        compactor.set_repo_map(repo_map)
        compactor.set_architectural_plan(plan)
        compactor.add_user_message(f"Run verification command: {test_command}\n\nContext:{file_context}")
        
        tools = TOOL_SCHEMAS
        model = self._get_model_for_role("verifier")
        
        final_content = ""
        last_exit_code = None
        
        for _ in range(10):  # Max tool rounds for verifier
            budget = compactor.check_token_budget(2000)
            if not budget["within_budget"]:
                compactor.compact()
            
            messages = self._merge_system_messages(compactor.build_messages(include_reasoning=False))
            
            content, reasoning = "", ""
            pending: Dict[int, Dict[str, str]] = {}
            
            async for chunk in self._nim_client.chat_completion_stream(
                model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.1
            ):
                data = self._parse_stream_chunk(chunk)
                if "error" in data:
                    raise RuntimeError(f"NIM error: {data['error']}")
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content += delta["content"]
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
            
            calls = [pending[i] for i in sorted(pending) if pending[i]["name"]]
            for i, c in enumerate(calls):
                c["id"] = c["id"] or f"call_{i}"
            
            if not calls:
                compactor.add_assistant_message(content, reasoning=reasoning)
                final_content = content
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
                except ValueError:
                    args = {}
                    output = f"Invalid tool arguments for {c['name']}"
                else:
                    output = await asyncio.to_thread(
                        lambda: asyncio.run(execute_tool_call(self.sandbox, c["name"], args))
                    )
                    if c["name"] == "run_terminal":
                        for line in output.splitlines():
                            if line.startswith("Exit code:"):
                                try:
                                    last_exit_code = int(line.split(":")[1].strip())
                                except Exception:
                                    pass
                
                compactor.add_tool_result(c["id"], c["name"], output)
        
        success = last_exit_code == 0 if last_exit_code is not None else (
            "passed" in final_content.lower() and "fail" not in final_content.lower() and "error" not in final_content.lower()
        )
        
        return SubAgentResult(
            success=success,
            summary=final_content,
            metadata={"exit_code": last_exit_code, "command": test_command}
        )
    
    def _read_context_files(self, context_files: List[str]) -> str:
        context = ""
        for f in context_files:
            try:
                content = self.sandbox.read_file(f)
                if len(content) > 20000:
                    content = content[:20000] + "\n... [truncated]"
                context += f"\n--- File: {f} ---\n{content}\n"
            except Exception as e:
                context += f"\n--- File: {f} (ERROR: {e}) ---\n"
        return context


# ==================== Factory Functions ====================

def create_explorer_subagent(
    sandbox: SandboxManager,
    settings_manager: SettingsManager,
    repo_mapper: Optional[RepoMapper] = None,
) -> ExplorerSubAgent:
    """Factory function to create an ExplorerSubAgent."""
    return ExplorerSubAgent(sandbox, settings_manager, repo_mapper)


def create_coder_subagent(
    sandbox: SandboxManager,
    settings_manager: SettingsManager,
    repo_mapper: Optional[RepoMapper] = None,
) -> CoderSubAgent:
    """Factory function to create a CoderSubAgent."""
    return CoderSubAgent(sandbox, settings_manager, repo_mapper)


def create_test_writer_subagent(
    sandbox: SandboxManager,
    settings_manager: SettingsManager,
    repo_mapper: Optional[RepoMapper] = None,
) -> TestWriterSubAgent:
    """Factory function to create a TestWriterSubAgent."""
    return TestWriterSubAgent(sandbox, settings_manager, repo_mapper)


def create_subagent_manager(
    sandbox: SandboxManager,
    settings_manager: SettingsManager,
    repo_mapper: Optional[RepoMapper] = None,
    max_workers: int = 2,
) -> SubAgentManager:
    """Factory function to create a SubAgentManager."""
    return SubAgentManager(sandbox, settings_manager, repo_mapper, max_workers)


def create_subagent_harness(
    sandbox: SandboxManager,
    settings_manager: SettingsManager,
    max_iterations: int = 5,
    git_guard=None,
) -> SubAgentHarness:
    """Factory function to create a SubAgentHarness."""
    return SubAgentHarness(sandbox, settings_manager, max_iterations, git_guard)