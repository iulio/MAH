"""
Multi-Agent Harness - Orchestrates the Planner → Coder → Verifier loop.

Implements two modes:
1. Ask/Chat Mode: Read-only codebase Q&A using list_files and read_file
2. Agent/Codex Mode: Full autonomous execution with planning, editing, and self-healing
"""

import json
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from config import NIMClient, NIMSettings, get_settings, SettingsManager
from sandbox import WorkspaceSandbox, TOOL_SCHEMAS, execute_tool_call, TerminalResult

logger = logging.getLogger(__name__)


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
    reasoning_content: Optional[str] = None  # For models that support reasoning


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


class AgentHarness:
    """
    Multi-agent harness for code generation and editing.
    
    Coordinates three specialized models:
    - Planner: Analyzes task and creates step-by-step plan
    - Coder: Implements changes via file edits
    - Verifier: Runs tests and verifies correctness
    """

    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        settings_manager: SettingsManager,
        mode: AgentMode = AgentMode.ASK,
        max_iterations: int = 5
    ):
        self.sandbox = sandbox
        self.settings_manager = settings_manager
        self.mode = mode
        self.max_iterations = max_iterations
        self._nim_client: Optional[NIMClient] = None
        self._openai_client: Optional[AsyncOpenAI] = None
        self._conversation: List[AgentMessage] = []
        self._steps: List[AgentStep] = []

    async def __aenter__(self) -> "AgentHarness":
        settings = self.settings_manager.settings
        nim_settings = settings.nim
        
        self._nim_client = NIMClient(nim_settings)
        await self._nim_client.__aenter__()
        
        # Also create OpenAI client for convenience
        self._openai_client = AsyncOpenAI(
            api_key=nim_settings.api_key.get_secret_value(),
            base_url=nim_settings.base_url,
            timeout=nim_settings.timeout
        )
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._nim_client:
            await self._nim_client.__aexit__(exc_type, exc_val, exc_tb)

    def _get_model_for_role(self, role: str) -> str:
        """Get the model assigned to a specific role."""
        models = self.settings_manager.settings.models
        if role == "planner":
            return models.planner_model
        elif role == "coder":
            return models.coder_model
        elif role == "verifier":
            return models.verifier_model
        return models.coder_model

    def _get_tools_for_mode(self) -> List[ChatCompletionToolParam]:
        """Get available tools based on current mode."""
        if self.mode == AgentMode.ASK:
            # Read-only tools for Ask mode
            return [t for t in TOOL_SCHEMAS if t["function"]["name"] in ("list_files", "read_file")]
        else:
            # All tools for Agent mode
            return TOOL_SCHEMAS

    def _build_system_prompt(self, role: str) -> str:
        """Build system prompt for a specific agent role."""
        base_prompt = """You are an expert software engineer working in a multi-agent coding system.
You have access to a workspace sandbox with file operations and terminal execution.
Always be precise, concise, and follow best practices."""

        if role == "planner":
            return base_prompt + """

ROLE: PLANNER
Your job is to analyze the user's request and the workspace context, then create a detailed, step-by-step execution plan.

Guidelines:
- Break down complex tasks into small, atomic steps
- Identify which files need to be read, created, or modified
- Specify the exact changes needed for each file
- Consider edge cases and potential issues
- Output a clear, numbered plan that the Coder can follow
- Do NOT execute any tools - only output the plan as text"""

        elif role == "coder":
            return base_prompt + """

ROLE: CODER
Your job is to implement the plan by making precise file edits.

Guidelines:
- Use apply_edit for surgical changes (provide exact search_block and replace_block)
- Use create_file for new files
- Make minimal, focused changes
- Follow the existing code style and patterns
- If you need to see a file first, use read_file
- Always verify your changes make sense in context
- Output tool calls to execute the changes"""

        elif role == "verifier":
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

    async def _stream_chat_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        tools: Optional[List[ChatCompletionToolParam]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.1
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from NIM."""
        if not self._nim_client:
            raise RuntimeError("NIM client not initialized")
        
        async for chunk in self._nim_client.chat_completion_stream(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature
        ):
            yield chunk

    async def _chat_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        tools: Optional[List[ChatCompletionToolParam]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Non-streaming chat completion."""
        if not self._nim_client:
            raise RuntimeError("NIM client not initialized")
        
        response = await self._nim_client.chat_completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature
        )
        return response.json()

    def _parse_stream_chunk(self, chunk: str) -> Dict[str, Any]:
        """Parse a streaming chunk from NIM."""
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return {}

    def _extract_tool_calls(self, message: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from a message."""
        tool_calls = []
        for tc in message.get("tool_calls", []):
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"])
            ))
        return tool_calls

    def _extract_reasoning(self, message: Dict[str, Any]) -> str:
        """Extract reasoning content from a message (for models that support it)."""
        return message.get("reasoning_content", "") or message.get("reasoning", "")

    async def _run_planner(self, user_prompt: str, context_files: List[str] = None) -> AsyncGenerator[str, None]:
        """Run the planner to create an execution plan. Yields streaming content."""
        model = self._get_model_for_role("planner")
        
        # Build context from mentioned files
        context = ""
        if context_files:
            for f in context_files:
                try:
                    content = self.sandbox.read_file(f)
                    context += f"\n--- File: {f} ---\n{content}\n"
                except Exception as e:
                    context += f"\n--- File: {f} (ERROR: {e}) ---\n"
        
        system_prompt = self._build_system_prompt("planner")
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {user_prompt}\n\nWorkspace Context:{context}"}
        ]
        
        # Stream the plan
        full_response = ""
        reasoning = ""
        tools = self._get_tools_for_mode()
        
        async for chunk in self._stream_chat_completion(model, messages, tools=tools):
            data = self._parse_stream_chunk(chunk)
            if "choices" in data and data["choices"]:
                delta = data["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    full_response += delta["content"]
                    yield delta["content"]
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    reasoning += delta["reasoning_content"]
        
        step = AgentStep(
            step_type="plan",
            model=model,
            input_messages=[AgentMessage(role=m["role"], content=m.get("content", "")) for m in messages],
            output=full_response,
            reasoning=reasoning
        )
        self._steps.append(step)
        # Yield a special marker with the step
        yield f"__STEP_COMPLETE__{json.dumps(step.__dict__, default=str)}"

    async def _run_coder(self, plan: str, context_files: List[str] = None) -> AgentStep:
        """Run the coder to implement the plan."""
        model = self._get_model_for_role("coder")
        
        # Build context
        context = f"Plan:\n{plan}\n\n"
        if context_files:
            for f in context_files:
                try:
                    content = self.sandbox.read_file(f)
                    context += f"\n--- File: {f} ---\n{content}\n"
                except Exception as e:
                    context += f"\n--- File: {f} (ERROR: {e}) ---\n"
        
        system_prompt = self._build_system_prompt("coder")
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        tools = self._get_tools_for_mode()
        tool_calls_made = []
        full_response = ""
        reasoning = ""
        
        # We need to handle tool calls in a loop
        while True:
            async for chunk in self._stream_chat_completion(model, messages, tools=tools):
                data = self._parse_stream_chunk(chunk)
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        full_response += delta["content"]
                        yield delta["content"]
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        reasoning += delta["reasoning_content"]
                    if "tool_calls" in delta:
                        # Accumulate tool calls
                        for tc in delta["tool_calls"]:
                            if len(tool_calls_made) <= tc["index"]:
                                tool_calls_made.append({"id": "", "name": "", "arguments": ""})
                            if "id" in tc:
                                tool_calls_made[tc["index"]]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    tool_calls_made[tc["index"]]["name"] = tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_calls_made[tc["index"]]["arguments"] += tc["function"]["arguments"]
            
            # Check if we have complete tool calls to execute
            if tool_calls_made and all(tc["id"] and tc["name"] for tc in tool_calls_made):
                # Execute tool calls
                for tc in tool_calls_made:
                    try:
                        args = json.loads(tc["arguments"])
                        result = await execute_tool_call(self.sandbox, tc["name"], args)
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": result
                        })
                        yield f"\n[Tool: {tc['name']}]\n{result}\n"
                    except Exception as e:
                        error_msg = f"Tool execution error: {e}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": error_msg
                        })
                        yield f"\n[Tool Error: {tc['name']}]\n{error_msg}\n"
                
                tool_calls_made = []
                continue  # Continue the loop for next response
            
            # Check if response is complete (no tool calls)
            if not tool_calls_made or not any(tc["name"] for tc in tool_calls_made):
                break
        
        step = AgentStep(
            step_type="edit",
            model=model,
            input_messages=[AgentMessage(role=m["role"], content=m.get("content", "")) for m in messages],
            output=full_response,
            tool_calls=[ToolCall(id=tc["id"], name=tc["name"], arguments=json.loads(tc["arguments"])) 
                       for tc in tool_calls_made if tc["id"]],
            reasoning=reasoning
        )
        self._steps.append(step)
        # Yield a special marker with the step
        yield f"__STEP_COMPLETE__{json.dumps(step.__dict__, default=str)}"

    async def _run_verifier(self, test_command: str = "pytest -v") -> AsyncGenerator[str, None]:
        """Run the verifier to test the implementation. Yields streaming content."""
        model = self._get_model_for_role("verifier")
        
        system_prompt = self._build_system_prompt("verifier")
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Run verification command: {test_command}"}
        ]
        
        tools = self._get_tools_for_mode()
        tool_calls_made = []
        full_response = ""
        reasoning = ""
        
        while True:
            async for chunk in self._stream_chat_completion(model, messages, tools=tools):
                data = self._parse_stream_chunk(chunk)
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        full_response += delta["content"]
                        yield delta["content"]
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        reasoning += delta["reasoning_content"]
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            if len(tool_calls_made) <= tc["index"]:
                                tool_calls_made.append({"id": "", "name": "", "arguments": ""})
                            if "id" in tc:
                                tool_calls_made[tc["index"]]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    tool_calls_made[tc["index"]]["name"] = tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_calls_made[tc["index"]]["arguments"] += tc["function"]["arguments"]
            
            if tool_calls_made and all(tc["id"] and tc["name"] for tc in tool_calls_made):
                for tc in tool_calls_made:
                    try:
                        args = json.loads(tc["arguments"])
                        result = await execute_tool_call(self.sandbox, tc["name"], args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": result
                        })
                        yield f"\n[Tool: {tc['name']}]\n{result}\n"
                    except Exception as e:
                        error_msg = f"Tool execution error: {e}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": error_msg
                        })
                        yield f"\n[Tool Error: {tc['name']}]\n{error_msg}\n"
                
                tool_calls_made = []
                continue
            
            if not tool_calls_made or not any(tc["name"] for tc in tool_calls_made):
                break
        
        step = AgentStep(
            step_type="verify",
            model=model,
            input_messages=[AgentMessage(role=m["role"], content=m.get("content", "")) for m in messages],
            output=full_response,
            tool_calls=[ToolCall(id=tc["id"], name=tc["name"], arguments=json.loads(tc["arguments"])) 
                       for tc in tool_calls_made if tc["id"]],
            reasoning=reasoning
        )
        self._steps.append(step)
        # Yield a special marker with the step
        yield f"__STEP_COMPLETE__{json.dumps(step.__dict__, default=str)}"

    async def _run_fixer(self, error_output: str, context_files: List[str] = None) -> AgentStep:
        """Run the coder to fix errors from verification."""
        model = self._get_model_for_role("coder")
        
        context = f"Previous verification failed with errors:\n{error_output}\n\n"
        context += "Please fix the issues and re-run verification.\n\n"
        
        if context_files:
            for f in context_files:
                try:
                    content = self.sandbox.read_file(f)
                    context += f"\n--- File: {f} ---\n{content}\n"
                except Exception as e:
                    context += f"\n--- File: {f} (ERROR: {e}) ---\n"
        
        system_prompt = self._build_system_prompt("coder") + "\n\nYou are now in FIX mode. Analyze the errors and apply corrections."
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        tools = self._get_tools_for_mode()
        tool_calls_made = []
        full_response = ""
        reasoning = ""
        
        while True:
            async for chunk in self._stream_chat_completion(model, messages, tools=tools):
                data = self._parse_stream_chunk(chunk)
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        full_response += delta["content"]
                        yield delta["content"]
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        reasoning += delta["reasoning_content"]
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            if len(tool_calls_made) <= tc["index"]:
                                tool_calls_made.append({"id": "", "name": "", "arguments": ""})
                            if "id" in tc:
                                tool_calls_made[tc["index"]]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    tool_calls_made[tc["index"]]["name"] = tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_calls_made[tc["index"]]["arguments"] += tc["function"]["arguments"]
            
            if tool_calls_made and all(tc["id"] and tc["name"] for tc in tool_calls_made):
                for tc in tool_calls_made:
                    try:
                        args = json.loads(tc["arguments"])
                        result = await execute_tool_call(self.sandbox, tc["name"], args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": result
                        })
                        yield f"\n[Tool: {tc['name']}]\n{result}\n"
                    except Exception as e:
                        error_msg = f"Tool execution error: {e}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": error_msg
                        })
                        yield f"\n[Tool Error: {tc['name']}]\n{error_msg}\n"
                
                tool_calls_made = []
                continue
            
            if not tool_calls_made or not any(tc["name"] for tc in tool_calls_made):
                break
        
        step = AgentStep(
            step_type="fix",
            model=model,
            input_messages=[AgentMessage(role=m["role"], content=m.get("content", "")) for m in messages],
            output=full_response,
            tool_calls=[ToolCall(id=tc["id"], name=tc["name"], arguments=json.loads(tc["arguments"])) 
                       for tc in tool_calls_made if tc["id"]],
            reasoning=reasoning
        )
        self._steps.append(step)
        # Yield a special marker with the step
        yield f"__STEP_COMPLETE__{json.dumps(step.__dict__, default=str)}"

    async def run_ask_mode(
        self,
        user_prompt: str,
        context_files: List[str] = None
    ) -> AsyncGenerator[str, None]:
        """Run in Ask/Chat mode - read-only Q&A."""
        model = self._get_model_for_role("coder")  # Use coder model for Q&A
        
        context = ""
        if context_files:
            for f in context_files:
                try:
                    content = self.sandbox.read_file(f)
                    context += f"\n--- File: {f} ---\n{content}\n"
                except Exception as e:
                    context += f"\n--- File: {f} (ERROR: {e}) ---\n"
        
        system_prompt = """You are an expert software engineer. Answer questions about the codebase using the provided context.
You can use list_files and read_file tools to explore the workspace further if needed.
Be concise and accurate."""
        
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {user_prompt}\n\nContext:{context}"}
        ]
        
        tools = self._get_tools_for_mode()
        tool_calls_made = []
        
        while True:
            async for chunk in self._stream_chat_completion(model, messages, tools=tools):
                data = self._parse_stream_chunk(chunk)
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        yield delta["content"]
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            if len(tool_calls_made) <= tc["index"]:
                                tool_calls_made.append({"id": "", "name": "", "arguments": ""})
                            if "id" in tc:
                                tool_calls_made[tc["index"]]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    tool_calls_made[tc["index"]]["name"] = tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_calls_made[tc["index"]]["arguments"] += tc["function"]["arguments"]
            
            if tool_calls_made and all(tc["id"] and tc["name"] for tc in tool_calls_made):
                for tc in tool_calls_made:
                    try:
                        args = json.loads(tc["arguments"])
                        result = await execute_tool_call(self.sandbox, tc["name"], args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": result
                        })
                        yield f"\n[Tool: {tc['name']}]\n{result}\n"
                    except Exception as e:
                        error_msg = f"Tool execution error: {e}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": error_msg
                        })
                        yield f"\n[Tool Error: {tc['name']}]\n{error_msg}\n"
                
                tool_calls_made = []
                continue
            
            if not tool_calls_made or not any(tc["name"] for tc in tool_calls_made):
                break

    async def run_agent_mode(
        self,
        user_prompt: str,
        context_files: List[str] = None,
        test_command: str = "pytest -v"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run full Agent/Codex mode: Plan → Edit → Verify → (Fix loop).
        
        Yields progress updates with step information.
        """
        self._steps = []
        iteration = 0
        
        # Step 1: Plan
        yield {"type": "step_start", "step": "plan", "iteration": iteration}
        plan_chunks = []
        async for chunk in self._run_planner(user_prompt, context_files):
            if chunk.startswith("__STEP_COMPLETE__"):
                # Parse the step from the marker
                step_data = json.loads(chunk[len("__STEP_COMPLETE__"):])
                # Step is already added to self._steps in _run_planner
                continue
            plan_chunks.append(chunk)
            yield {"type": "stream", "step": "plan", "content": chunk}
        plan = "".join(plan_chunks)
        yield {"type": "step_complete", "step": "plan", "content": plan}
        
        # Step 2: Edit
        yield {"type": "step_start", "step": "edit", "iteration": iteration}
        edit_chunks = []
        async for chunk in self._run_coder(plan, context_files):
            if chunk.startswith("__STEP_COMPLETE__"):
                step_data = json.loads(chunk[len("__STEP_COMPLETE__"):])
                continue
            edit_chunks.append(chunk)
            yield {"type": "stream", "step": "edit", "content": chunk}
        yield {"type": "step_complete", "step": "edit", "content": "".join(edit_chunks)}
        
        # Step 3: Verify + Self-heal loop
        while iteration < self.max_iterations:
            yield {"type": "step_start", "step": "verify", "iteration": iteration}
            verify_chunks = []
            async for chunk in self._run_verifier(test_command):
                if chunk.startswith("__STEP_COMPLETE__"):
                    step_data = json.loads(chunk[len("__STEP_COMPLETE__"):])
                    continue
                verify_chunks.append(chunk)
                yield {"type": "stream", "step": "verify", "content": chunk}
            verify_output = "".join(verify_chunks)
            yield {"type": "step_complete", "step": "verify", "content": verify_output}
            
            # Check if verification passed
            last_step = self._steps[-1]
            # Simple heuristic: check if tests passed
            if "passed" in verify_output.lower() and "failed" not in verify_output.lower():
                yield {"type": "success", "message": "All tests passed!", "iterations": iteration + 1}
                break
            
            # If failed and we have iterations left, run fixer
            if iteration < self.max_iterations - 1:
                yield {"type": "step_start", "step": "fix", "iteration": iteration}
                fix_chunks = []
                async for chunk in self._run_fixer(verify_output, context_files):
                    if chunk.startswith("__STEP_COMPLETE__"):
                        step_data = json.loads(chunk[len("__STEP_COMPLETE__"):])
                        continue
                    fix_chunks.append(chunk)
                    yield {"type": "stream", "step": "fix", "content": chunk}
                yield {"type": "step_complete", "step": "fix", "content": "".join(fix_chunks)}
                iteration += 1
            else:
                yield {"type": "failure", "message": f"Max iterations ({self.max_iterations}) reached. Tests still failing.", "iterations": iteration + 1}
                break
        
        yield {"type": "done", "steps": self._steps}

    def get_conversation_history(self) -> List[AgentMessage]:
        """Get the conversation history."""
        return self._conversation

    def get_steps(self) -> List[AgentStep]:
        """Get execution steps."""
        return self._steps

    def clear_history(self) -> None:
        """Clear conversation history and steps."""
        self._conversation = []
        self._steps = []


async def create_harness(
    workspace_dir: str,
    mode: AgentMode = AgentMode.ASK,
    max_iterations: int = 5,
    auto_approve: bool = False,
    command_timeout: int = 120
) -> AgentHarness:
    """Factory function to create and initialize an AgentHarness."""
    settings_manager = get_settings_manager()
    sandbox = WorkspaceSandbox(workspace_dir, auto_approve, command_timeout)
    harness = AgentHarness(sandbox, settings_manager, mode, max_iterations)
    await harness.__aenter__()
    return harness