"""
Context Manager - Sliding-window context compaction for token-efficient conversations.

Manages multi-turn agent trajectory:
- Keeps system prompt, repo_map, architectural plan, last N tool turns in full detail
- Compresses older read_file contents and verbose run_bash logs into 2-line summaries
- Token estimation via tiktoken (fallback: character heuristic)
- Pre-flight token check before each NIM API call
"""

import json
import re
from typing import List, Dict, Any, Optional, Literal, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    REASONING = "reasoning"


class ToolType(str, Enum):
    """Types of tool calls for compression strategy."""
    READ_FILE = "read_file"
    RUN_TERMINAL = "run_terminal"
    LIST_FILES = "list_files"
    APPLY_EDIT = "apply_edit"
    CREATE_FILE = "create_file"
    OTHER = "other"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: MessageRole
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    original_tokens: int = 0
    compressed_tokens: int = 0


@dataclass
class ContextWindow:
    """Managed context window with token tracking."""
    system_prompt: str = ""
    repo_map: str = ""
    architectural_plan: str = ""
    turns: List[ConversationTurn] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 8000


class TokenEstimator:
    """Token estimation with tiktoken fallback to heuristic."""
    
    def __init__(self, method: Literal["tiktoken", "heuristic"] = "tiktoken", encoding: str = "cl100k_base"):
        self.method = method
        self.encoding_name = encoding
        self._encoder = None
        
        if method == "tiktoken" and TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(encoding)
            except Exception:
                self.method = "heuristic"
    
    def estimate(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        
        if self.method == "tiktoken" and self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        
        # Heuristic: ~4 characters per token for English/code
        return len(text) // 4
    
    def estimate_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate tokens for a list of chat messages."""
        total = 0
        for msg in messages:
            # Role overhead
            total += 4
            # Content
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate(content)
            # Tool calls
            for tc in msg.get("tool_calls", []):
                total += self.estimate(json.dumps(tc))
        return total


class ContextCompactor:
    """
    Sliding-window context compactor for multi-turn agent conversations.
    
    Strategy:
    1. Always keep: system prompt, repo_map, architectural plan
    2. Keep last N tool turns (default 3) in full detail
    3. Compress older turns:
       - read_file: "Read file.py (150 lines) → [summary]"
       - run_terminal: "pytest -v → Exit 1: 3 failed, 12 passed"
       - list_files: "Listed src/ → 23 files"
       - apply_edit/create_file: Keep full (they're mutations)
    4. Pre-flight token check before API calls
    """
    
    def __init__(
        self,
        max_tokens: int = 8000,
        keep_last_turns: int = 3,
        summary_max_lines: int = 2,
        token_method: Literal["tiktoken", "heuristic"] = "tiktoken"
    ):
        self.max_tokens = max_tokens
        self.keep_last_turns = keep_last_turns
        self.summary_max_lines = summary_max_lines
        self.token_estimator = TokenEstimator(token_method)
        self.window = ContextWindow(max_tokens=max_tokens)
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt (always kept in full)."""
        self.window.system_prompt = prompt
        self._recalculate_tokens()
    
    def set_repo_map(self, repo_map: str) -> None:
        """Set the repository map (always kept in full)."""
        self.window.repo_map = repo_map
        self._recalculate_tokens()
    
    def set_architectural_plan(self, plan: str) -> None:
        """Set the architectural plan (always kept in full)."""
        self.window.architectural_plan = plan
        self._recalculate_tokens()
    
    def add_turn(
        self,
        role: MessageRole,
        content: str,
        tool_calls: List[Dict[str, Any]] = None,
        tool_results: List[Dict[str, Any]] = None,
        reasoning: str = "",
        metadata: Dict[str, Any] = None
    ) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            reasoning=reasoning,
            metadata=metadata or {}
        )
        
        # Estimate tokens for this turn
        turn_text = self._turn_to_text(turn)
        turn.original_tokens = self.token_estimator.estimate(turn_text)
        
        self.window.turns.append(turn)
        self._recalculate_tokens()
        
        # Auto-compact if over budget
        if self.window.total_tokens > self.max_tokens:
            self.compact()
        
        return turn
    
    def add_user_message(self, content: str, metadata: Dict = None) -> ConversationTurn:
        """Add a user message."""
        return self.add_turn(MessageRole.USER, content, metadata=metadata)
    
    def add_assistant_message(
        self,
        content: str,
        tool_calls: List[Dict] = None,
        reasoning: str = "",
        metadata: Dict = None
    ) -> ConversationTurn:
        """Add an assistant message with optional tool calls."""
        return self.add_turn(
            MessageRole.ASSISTANT,
            content,
            tool_calls=tool_calls,
            reasoning=reasoning,
            metadata=metadata
        )
    
    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        metadata: Dict = None
    ) -> ConversationTurn:
        """Add a tool result (typically paired with previous assistant turn)."""
        # Find the last assistant turn with matching tool call
        for turn in reversed(self.window.turns):
            if turn.role == MessageRole.ASSISTANT:
                for tc in turn.tool_calls:
                    if tc.get("id") == tool_call_id:
                        turn.tool_results.append({
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": result
                        })
                        # Recalculate tokens for this turn
                        turn_text = self._turn_to_text(turn)
                        turn.original_tokens = self.token_estimator.estimate(turn_text)
                        self._recalculate_tokens()
                        return turn
        
        # If not found, create a new tool turn
        return self.add_turn(
            MessageRole.TOOL,
            result,
            metadata={"tool_call_id": tool_call_id, "tool_name": tool_name, **(metadata or {})}
        )
    
    def _turn_to_text(self, turn: ConversationTurn) -> str:
        """Convert turn to text for token estimation (includes tool args and results)."""
        parts = [turn.content or ""]
        if turn.reasoning:
            parts.append(f"[Reasoning: {turn.reasoning}]")
        for tc in turn.tool_calls:
            name, args = self._tool_name_args(tc)
            parts.append(f"[Tool: {name}] {json.dumps(args)}")
        for tr in turn.tool_results:
            parts.append(f"[Result: {tr.get('name', 'unknown')}] {tr.get('content', '')}")
        return "\n".join(parts)

    @staticmethod
    def _tool_name_args(tc: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract (name, args dict) from an OpenAI-format or flat tool call."""
        func = tc.get("function", tc)
        name = func.get("name", "") or ""
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return name, args if isinstance(args, dict) else {}

    def _tool_turns(self) -> List[ConversationTurn]:
        """Turns that carry tool traffic (the unit of 'last N turns')."""
        return [
            t for t in self.window.turns
            if t.tool_calls or t.tool_results or t.role == MessageRole.TOOL
        ]
    
    def _recalculate_tokens(self) -> None:
        """Recalculate total token count."""
        total = 0
        total += self.token_estimator.estimate(self.window.system_prompt)
        total += self.token_estimator.estimate(self.window.repo_map)
        total += self.token_estimator.estimate(self.window.architectural_plan)
        
        for turn in self.window.turns:
            if turn.compressed:
                total += turn.compressed_tokens
            else:
                total += turn.original_tokens
        
        self.window.total_tokens = total
    
    def _compress_candidates(self) -> List[ConversationTurn]:
        """Tool turns older than the last `keep_last_turns` that are not yet compressed."""
        tool_turns = self._tool_turns()
        older = tool_turns[:-self.keep_last_turns] if self.keep_last_turns > 0 else tool_turns
        return [t for t in older if not t.compressed]

    def _apply_compression(self, turn: ConversationTurn) -> None:
        compressed_text = self._compress_turn(turn)
        turn.compressed_tokens = self.token_estimator.estimate(compressed_text)
        turn.content = compressed_text
        turn.tool_calls = []
        turn.tool_results = []
        turn.reasoning = ""
        turn.role = MessageRole.ASSISTANT
        turn.compressed = True

    def compact(self) -> int:
        """
        Compress older tool turns (read_file / run_terminal / list_files output) into
        short summaries. User messages, the plan and the last N tool turns stay intact.

        Returns:
            Number of turns compressed
        """
        candidates = self._compress_candidates()
        for turn in candidates:
            self._apply_compression(turn)
        self._recalculate_tokens()
        return len(candidates)

    def _compress_turn(self, turn: ConversationTurn) -> str:
        """Compress a tool turn into a short (<= summary_max_lines) summary."""
        pairs: List[Tuple[str, Dict[str, Any], str]] = []
        results_by_id = {tr.get("tool_call_id"): tr.get("content", "") for tr in turn.tool_results}
        for tc in turn.tool_calls:
            name, args = self._tool_name_args(tc)
            pairs.append((name, args, results_by_id.get(tc.get("id"), "")))
        if not pairs and turn.tool_results:
            tr = turn.tool_results[0]
            pairs.append((tr.get("name", ""), {}, tr.get("content", "")))

        lines: List[str] = []
        for name, args, result in pairs:
            if name == "read_file":
                lines.append(self._summarize_read_file(args, result))
            elif name in ("run_terminal", "run_bash"):
                lines.append(self._summarize_terminal(args, result))
            elif name == "list_files":
                lines.append(self._summarize_list_files(result))
            elif name == "apply_edit":
                first = result.splitlines()[0] if result else "ok"
                lines.append(f"[apply_edit {args.get('path', '?')} → {first[:100]}]")
            elif name == "create_file":
                lines.append(f"[create_file {args.get('path', '?')} → {result[:100]}]")
            else:
                lines.append(f"[{name or 'tool'} → {result[:100]}]")

        if not lines:
            return self._summarize_generic(turn)
        return "\n".join(lines[:max(1, self.summary_max_lines)])
    
    def _summarize_read_file(self, args: Dict, result: str) -> str:
        """Summarize a read_file operation."""
        path = args.get("path", "unknown")
        start = args.get("start_line", 1)
        end = args.get("end_line")
        
        lines_read = "full file"
        if end:
            lines_read = f"lines {start}-{end}"
        elif start > 1:
            lines_read = f"from line {start}"
        
        # Extract key info from result
        result_lines = result.splitlines()
        line_count = len(result_lines)
        
        # Try to extract first few meaningful lines
        preview = ""
        for line in result_lines[:3]:
            if line.strip() and not line.strip().startswith("#"):
                preview = line.strip()[:100]
                break
        
        summary = f"[Read {path} ({lines_read}, {line_count} lines)"
        if preview:
            summary += f" → {preview}"
        summary += "]"
        
        return summary
    
    def _summarize_terminal(self, args: Dict, result: str) -> str:
        """Summarize a run_terminal operation."""
        command = args.get("command", "unknown")
        
        # Extract exit code and key output
        exit_code = 0
        stdout = ""
        stderr = ""
        
        for line in result.splitlines():
            if line.startswith("Exit code:"):
                try:
                    exit_code = int(line.split(":")[1].strip())
                except Exception:
                    pass
            elif line.startswith("STDOUT:"):
                stdout = result.split("STDOUT:")[1].split("STDERR:")[0].strip()
            elif line.startswith("STDERR:"):
                stderr = result.split("STDERR:")[1].split("Duration:")[0].strip()
        
        # Build summary
        status = "✓" if exit_code == 0 else f"✗ (exit {exit_code})"
        summary = f"[Terminal: {command} → {status}]"
        
        # Add key output (first/last few lines)
        if stdout:
            stdout_lines = [l for l in stdout.splitlines() if l.strip()]
            if stdout_lines:
                summary += f" stdout: {stdout_lines[0][:80]}"
                if len(stdout_lines) > 1:
                    summary += f" ... {stdout_lines[-1][:80]}"
        
        if stderr and exit_code != 0:
            stderr_lines = [l for l in stderr.splitlines() if l.strip()]
            if stderr_lines:
                summary += f" stderr: {stderr_lines[0][:80]}"
        
        return summary
    
    def _summarize_list_files(self, result: str) -> str:
        """Summarize a list_files operation."""
        lines = result.splitlines()
        file_count = sum(1 for l in lines if l.strip() and not l.startswith(" "))
        return f"[Listed files → {file_count} entries]"
    
    def _summarize_generic(self, turn: ConversationTurn) -> str:
        """Generic summarization for other content."""
        content = turn.content
        if len(content) > 200:
            content = content[:200] + "..."
        return f"[Compressed: {content}]"
    
    def build_messages(self, include_reasoning: bool = True) -> List[Dict[str, Any]]:
        """
        Build message list for API call with current context window.
        
        Args:
            include_reasoning: Whether to include reasoning blocks
            
        Returns:
            List of messages ready for chat completion API
        """
        messages = []
        
        # System prompt (always included)
        if self.window.system_prompt:
            messages.append({"role": "system", "content": self.window.system_prompt})
        
        # Repo map (always included)
        if self.window.repo_map:
            messages.append({"role": "system", "content": self.window.repo_map})
        
        # Architectural plan (always included)
        if self.window.architectural_plan:
            messages.append({"role": "system", "content": f"Architectural Plan:\n{self.window.architectural_plan}"})
        
        # Conversation turns
        for turn in self.window.turns:
            if turn.role == MessageRole.SYSTEM:
                continue  # Already handled above

            if turn.role == MessageRole.TOOL:
                messages.append({
                    "role": "tool",
                    "tool_call_id": turn.metadata.get("tool_call_id", ""),
                    "name": turn.metadata.get("tool_name", ""),
                    "content": turn.content
                })
                continue

            msg = {"role": turn.role.value, "content": turn.content or ""}
            if turn.tool_calls:
                msg["tool_calls"] = turn.tool_calls
            if include_reasoning and turn.reasoning:
                msg["reasoning_content"] = turn.reasoning
            messages.append(msg)

            # Tool results follow the assistant message that requested them
            for tr in turn.tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_call_id", ""),
                    "name": tr.get("name", ""),
                    "content": tr.get("content", "")
                })

        return messages
    
    def check_token_budget(self, estimated_response_tokens: int = 2000) -> Dict[str, Any]:
        """
        Pre-flight token budget check before API call.
        
        Args:
            estimated_response_tokens: Expected response token count
            
        Returns:
            Dict with budget status and recommendations
        """
        current = self.window.total_tokens
        projected = current + estimated_response_tokens
        
        return {
            "current_tokens": current,
            "max_tokens": self.max_tokens,
            "projected_tokens": projected,
            "within_budget": projected <= self.max_tokens,
            "headroom": self.max_tokens - projected,
            "recommendation": "compact" if projected > self.max_tokens else "ok"
        }
    
    def force_compact(self, target_tokens: Optional[int] = None) -> int:
        """
        Force compaction to a specific token target.
        
        Args:
            target_tokens: Target token count (defaults to 80% of max)
            
        Returns:
            Number of turns compressed
        """
        target = target_tokens or int(self.max_tokens * 0.8)
        
        if self.window.total_tokens <= target:
            return 0
        
        compressed = 0
        for turn in self._compress_candidates():
            self._apply_compression(turn)
            compressed += 1
            self._recalculate_tokens()
            if self.window.total_tokens <= target:
                break

        return compressed
    
    def truncate_large_results(self, max_chars: int = 6000) -> int:
        """
        Last-resort pre-flight guard: clip oversized tool results (even in the most recent
        turns) so a single huge file read or log cannot overflow the context window.

        Returns:
            Number of tool results truncated
        """
        truncated = 0
        for turn in self.window.turns:
            for tr in turn.tool_results:
                content = tr.get("content", "")
                if len(content) > max_chars:
                    head = content[: max_chars * 2 // 3]
                    tail = content[-max_chars // 3:]
                    omitted = len(content) - len(head) - len(tail)
                    tr["content"] = f"{head}\n... [{omitted} chars omitted to fit context] ...\n{tail}"
                    turn.original_tokens = self.token_estimator.estimate(self._turn_to_text(turn))
                    truncated += 1
        if truncated:
            self._recalculate_tokens()
        return truncated

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context window state."""
        return {
            "total_tokens": self.window.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization_pct": round(self.window.total_tokens / self.max_tokens * 100, 1),
            "total_turns": len(self.window.turns),
            "compressed_turns": sum(1 for t in self.window.turns if t.compressed),
            "full_detail_turns": sum(1 for t in self.window.turns if not t.compressed),
            "has_repo_map": bool(self.window.repo_map),
            "has_plan": bool(self.window.architectural_plan),
            "repo_map_tokens": self.token_estimator.estimate(self.window.repo_map),
            "plan_tokens": self.token_estimator.estimate(self.window.architectural_plan)
        }
    
    def export_for_debug(self) -> Dict[str, Any]:
        """Export full context for debugging."""
        return {
            "window": {
                "system_prompt": self.window.system_prompt[:200] + "..." if len(self.window.system_prompt) > 200 else self.window.system_prompt,
                "repo_map": self.window.repo_map[:200] + "..." if len(self.window.repo_map) > 200 else self.window.repo_map,
                "architectural_plan": self.window.architectural_plan[:200] + "..." if len(self.window.architectural_plan) > 200 else self.window.architectural_plan,
                "turns": [
                    {
                        "role": t.role.value,
                        "content_preview": t.content[:100] + "..." if len(t.content) > 100 else t.content,
                        "compressed": t.compressed,
                        "original_tokens": t.original_tokens,
                        "compressed_tokens": t.compressed_tokens,
                        "tool_calls": len(t.tool_calls),
                        "tool_results": len(t.tool_results)
                    }
                    for t in self.window.turns
                ]
            },
            "summary": self.get_context_summary()
        }


def create_context_compactor(
    max_tokens: int = 8000,
    keep_last_turns: int = 3,
    summary_max_lines: int = 2,
    token_method: Literal["tiktoken", "heuristic"] = "tiktoken"
) -> ContextCompactor:
    """Factory function to create a ContextCompactor instance."""
    return ContextCompactor(
        max_tokens=max_tokens,
        keep_last_turns=keep_last_turns,
        summary_max_lines=summary_max_lines,
        token_method=token_method
    )