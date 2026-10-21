"""
Sandbox Executor - Secure, isolated code execution environment.

Supports two execution modes:
1. Local: Direct subprocess execution on host
2. Docker: Persistent container with workspace mounted, commands via docker exec

Includes DiffEngine for fuzzy search-and-replace with syntax validation.
"""

import os
import subprocess
import difflib
import ast
import shlex
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import pathspec


class SandboxType(str, Enum):
    """Sandbox execution mode."""
    LOCAL = "local"
    DOCKER = "docker"


@dataclass
class FileEntry:
    """Represents a file or directory entry."""
    name: str
    path: str
    is_dir: bool
    size: int = 0
    children: List["FileEntry"] = field(default_factory=list)


@dataclass
class TerminalResult:
    """Result of a terminal command execution."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_ms: int = 0


@dataclass
class DiffResult:
    """Result of a diff/patch operation."""
    success: bool
    diff: str = ""
    error: str = ""
    syntax_error: Optional[str] = None
    new_content: Optional[str] = None


class DiffEngine:
    """
    Fuzzy search-and-replace patching engine with syntax validation.
    
    Features:
    - Tolerates minor whitespace/indentation drift using SequenceMatcher
    - Syntax validation for Python files via ast.parse() BEFORE writing
    - Returns detailed error information without saving broken files
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def find_best_match(self, content: str, search_block: str) -> Tuple[int, int, float]:
        """
        Find the best matching location for search_block in content.
        
        Returns:
            Tuple of (start_index, end_index, similarity_score)
            Returns (-1, -1, 0.0) if no match above threshold
        """
        start, end, score, _ = self._locate(content, search_block)
        return start, end, score

    def _locate(self, content: str, search_block: str) -> Tuple[int, int, float, bool]:
        """
        Locate search_block in content.

        Returns (start, end, score, ambiguous). Exact matches are tried first;
        otherwise a line-aligned sliding window is compared with
        indentation/trailing-whitespace stripped, so only whole lines are matched.
        """
        if not search_block.strip():
            return -1, -1, 0.0, False

        exact_count = content.count(search_block)
        if exact_count >= 1:
            idx = content.find(search_block)
            return idx, idx + len(search_block), 1.0, exact_count > 1

        content_lines = content.splitlines(keepends=True)
        search_norm = [l.strip() for l in search_block.strip("\r\n").splitlines()]
        n = len(search_norm)
        if n == 0 or len(content_lines) < n:
            return -1, -1, 0.0, False

        content_norm = [l.strip() for l in content_lines]
        offsets = [0]
        for line in content_lines:
            offsets.append(offsets[-1] + len(line))

        target = "\n".join(search_norm)
        best_score, best_i, best_count = 0.0, -1, 0
        for i in range(len(content_lines) - n + 1):
            candidate = "\n".join(content_norm[i:i + n])
            matcher = difflib.SequenceMatcher(None, candidate, target, autojunk=False)
            if matcher.real_quick_ratio() < max(best_score, self.similarity_threshold):
                continue
            score = matcher.ratio()
            if score > best_score + 1e-9:
                best_score, best_i, best_count = score, i, 1
            elif abs(score - best_score) <= 1e-9 and score >= self.similarity_threshold:
                best_count += 1

        if best_i == -1 or best_score < self.similarity_threshold:
            return -1, -1, best_score, False
        return offsets[best_i], offsets[best_i + n], best_score, best_count > 1

    @staticmethod
    def _leading_ws(text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line[:len(line) - len(line.lstrip())]
        return ""

    def _reindent(self, replace_block: str, search_block: str, matched_text: str) -> str:
        """Shift replace_block by the indentation drift between search_block and the matched text."""
        s_ind = self._leading_ws(search_block)
        m_ind = self._leading_ws(matched_text)
        if s_ind == m_ind:
            return replace_block

        lines = replace_block.splitlines(keepends=True)
        if m_ind.startswith(s_ind):
            extra = m_ind[len(s_ind):]
            return "".join(extra + l if l.strip() else l for l in lines)
        if s_ind.startswith(m_ind):
            drop = len(s_ind) - len(m_ind)
            return "".join(l[drop:] if l.strip() and l[:drop].strip() == "" else l for l in lines)
        return replace_block

    def validate_python_syntax(self, content: str, filepath: str) -> Optional[str]:
        """
        Validate Python syntax using ast.parse.
        
        Returns:
            None if valid, error message string if invalid
        """
        if not filepath.endswith('.py'):
            return None
        
        try:
            ast.parse(content, filename=filepath)
            return None
        except SyntaxError as e:
            return f"Syntax error in {filepath}:{e.lineno}: {e.msg}"
        except Exception as e:
            return f"Parse error in {filepath}: {str(e)}"

    def apply_edit(
        self,
        filepath: str,
        original_content: str,
        search_block: str,
        replace_block: str
    ) -> DiffResult:
        """
        Apply a surgical edit with fuzzy matching and syntax validation.
        
        Args:
            filepath: Path to the file (for syntax validation)
            original_content: Current file content
            search_block: Text to search for (exact or fuzzy)
            replace_block: Replacement text
            
        Returns:
            DiffResult with success status, diff, and any errors
        """
        start, end, similarity, ambiguous = self._locate(original_content, search_block)

        if start == -1:
            return DiffResult(
                success=False,
                error=f"Search block not found (best similarity: {similarity:.2%}, threshold: {self.similarity_threshold:.0%})",
                syntax_error=None
            )

        if ambiguous:
            return DiffResult(
                success=False,
                error="Search block matches multiple locations; include more surrounding context to make it unique"
            )

        matched_text = original_content[start:end]
        if matched_text != search_block:
            replace_block = self._reindent(replace_block, search_block, matched_text)
            if matched_text.endswith("\n") and replace_block and not replace_block.endswith("\n"):
                replace_block += "\n"

        new_content = original_content[:start] + replace_block + original_content[end:]

        # Syntax validation BEFORE writing
        syntax_error = self.validate_python_syntax(new_content, filepath)
        if syntax_error:
            return DiffResult(
                success=False,
                error="Syntax validation failed",
                syntax_error=syntax_error
            )

        # Generate unified diff
        diff = self._generate_unified_diff(
            original_content, new_content, filepath
        )

        return DiffResult(
            success=True,
            diff=diff,
            error="",
            syntax_error=None,
            new_content=new_content
        )

    def _generate_unified_diff(
        self,
        original: str,
        modified: str,
        filepath: str
    ) -> str:
        """Generate a unified diff between two strings."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm=""
        )

        return "".join(diff)


class SandboxManager:
    """
    Unified sandbox manager supporting both local and Docker execution.
    
    Docker Mode:
    - Starts a persistent container: `docker run -d -v workspace:/workspace python:3.12-slim sleep infinity`
    - Executes commands via `docker exec` with resource limits
    - Automatic container lifecycle management
    - Health checks and auto-restart
    
    Local Mode:
    - Direct subprocess execution with path validation
    - Same interface as Docker mode
    """

    def __init__(
        self,
        workspace_dir: str,
        sandbox_type: SandboxType = SandboxType.LOCAL,
        docker_image: str = "python:3.12-slim",
        docker_cpu_limit: str = "1.0",
        docker_memory_limit: str = "2g",
        command_timeout: int = 120,
        auto_approve: bool = False
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.sandbox_type = sandbox_type
        self.docker_image = docker_image
        self.docker_cpu_limit = docker_cpu_limit
        self.docker_memory_limit = docker_memory_limit
        self.command_timeout = command_timeout
        self.auto_approve = auto_approve
        
        self._docker_container_id: Optional[str] = None
        self._docker_container_name: str = f"nim-sandbox-{uuid.uuid4().hex[:8]}"
        self._gitignore_spec: Optional[pathspec.PathSpec] = None
        self._diff_engine = DiffEngine()
        
        self._load_gitignore()
        
        # Validate workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _load_gitignore(self) -> None:
        """Load .gitignore patterns for file listing."""
        gitignore_path = self.workspace_dir / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    patterns = f.read().splitlines()
                self._gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            except Exception:
                self._gitignore_spec = None

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored per .gitignore."""
        if self._gitignore_spec is None:
            return False
        try:
            rel_path = path.relative_to(self.workspace_dir)
            return self._gitignore_spec.match_file(str(rel_path))
        except Exception:
            return False

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve a path within the workspace.
        
        Raises:
            ValueError: If path is outside workspace or invalid.
        """
        target = (self.workspace_dir / path).resolve()
        
        try:
            target.relative_to(self.workspace_dir)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside workspace directory")
        
        return target

    # ==================== Docker Container Management ====================

    def _ensure_docker_container(self) -> bool:
        """
        Ensure Docker container is running.
        
        Returns:
            True if container is ready, False otherwise
        """
        if self.sandbox_type != SandboxType.DOCKER:
            return True

        # Check if container exists and is running
        if self._docker_container_id:
            if self._check_container_health():
                return True
            else:
                # Container unhealthy, try to restart
                self._stop_container()
                self._docker_container_id = None

        # Start new container
        return self._start_container()

    def _start_container(self) -> bool:
        """Start a new persistent Docker container."""
        try:
            # Build docker run command
            cmd = [
                "docker", "run", "-d",
                "--name", self._docker_container_name,
                "-v", f"{self.workspace_dir}:/workspace",
                "-w", "/workspace",
                "--cpus", self.docker_cpu_limit,
                "--memory", self.docker_memory_limit,
                self.docker_image,
                "sleep", "infinity"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                # Check if container already exists
                if "already in use" in result.stderr:
                    # Try to remove existing container
                    subprocess.run(["docker", "rm", "-f", self._docker_container_name], 
                                 capture_output=True)
                    # Retry
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    print(f"[SandboxManager] Failed to start Docker container: {result.stderr}")
                    return False

            self._docker_container_id = result.stdout.strip()
            
            # Wait for container to be ready
            time.sleep(1)
            
            if self._check_container_health():
                print(f"[SandboxManager] Docker container started: {self._docker_container_id[:12]}")
                return True
            else:
                print("[SandboxManager] Container started but health check failed")
                self._stop_container()
                return False

        except subprocess.TimeoutExpired:
            print("[SandboxManager] Timeout starting Docker container")
            return False
        except Exception as e:
            print(f"[SandboxManager] Error starting Docker container: {e}")
            return False

    def _check_container_health(self) -> bool:
        """Check if Docker container is healthy and responsive."""
        if not self._docker_container_id:
            return False
        
        try:
            result = subprocess.run(
                ["docker", "exec", self._docker_container_id, "echo", "healthcheck"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and "healthcheck" in result.stdout
        except Exception:
            return False

    def _stop_container(self) -> None:
        """Stop and remove the Docker container."""
        if self._docker_container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._docker_container_id],
                    capture_output=True,
                    timeout=10
                )
                subprocess.run(
                    ["docker", "rm", self._docker_container_id],
                    capture_output=True,
                    timeout=10
                )
                print(f"[SandboxManager] Docker container stopped: {self._docker_container_id[:12]}")
            except Exception as e:
                print(f"[SandboxManager] Error stopping container: {e}")
            finally:
                self._docker_container_id = None

    def start_container(self) -> bool:
        """Start (or verify) the persistent Docker container. No-op for local mode."""
        return self._ensure_docker_container()

    def stop_container(self) -> None:
        """Stop and remove the Docker container (no-op for local mode)."""
        if self.sandbox_type == SandboxType.DOCKER:
            self._stop_container()

    def get_container_status(self) -> Dict[str, Any]:
        """Get Docker container status for UI."""
        if self.sandbox_type != SandboxType.DOCKER:
            return {"type": "local", "status": "N/A", "container_id": None}
        
        if not self._docker_container_id:
            return {"type": "docker", "status": "stopped", "container_id": None}
        
        healthy = self._check_container_health()
        return {
            "type": "docker",
            "status": "running" if healthy else "unhealthy",
            "container_id": self._docker_container_id[:12] if self._docker_container_id else None,
            "image": self.docker_image,
            "cpu_limit": self.docker_cpu_limit,
            "memory_limit": self.docker_memory_limit
        }

    # ==================== File Operations ====================

    def list_files(self, directory: str = ".") -> str:
        """List files in a directory as a tree view."""
        target = self._validate_path(directory)
        
        if not target.exists():
            return f"Directory not found: {directory}"
        
        if not target.is_dir():
            return f"Not a directory: {directory}"
        
        entries = self._build_tree(target)
        return self._format_tree(entries, directory)

    def _build_tree(self, root: Path) -> List[FileEntry]:
        """Build file tree recursively."""
        entries = []
        
        try:
            items = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return entries
        
        for item in items:
            if self._should_ignore(item):
                continue
                
            if item.is_dir():
                children = self._build_tree(item)
                entries.append(FileEntry(
                    name=item.name + "/",
                    path=str(item.relative_to(self.workspace_dir)),
                    is_dir=True,
                    children=children
                ))
            else:
                try:
                    size = item.stat().st_size
                except Exception:
                    size = 0
                entries.append(FileEntry(
                    name=item.name,
                    path=str(item.relative_to(self.workspace_dir)),
                    is_dir=False,
                    size=size
                ))
        
        return entries

    def _format_tree(self, entries: List[FileEntry], root_name: str, prefix: str = "") -> str:
        """Format tree entries as a string."""
        lines = [f"{root_name}/"]
        
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            
            if entry.is_dir and entry.children:
                child_prefix = prefix + ("    " if is_last else "│   ")
                child_lines = self._format_tree(entry.children, "", child_prefix)
                lines.extend(child_lines.splitlines()[1:])
        
        return "\n".join(lines)

    def read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        """Read file contents with optional line range."""
        target = self._validate_path(path)
        
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not target.is_file():
            raise ValueError(f"Not a file: {path}")
        
        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        start_idx = max(0, start_line - 1)
        end_idx = len(lines) if end_line is None else min(len(lines), end_line)
        
        selected = lines[start_idx:end_idx]
        return "".join(selected)

    def create_file(self, path: str, content: str) -> str:
        """Create or overwrite a file."""
        target = self._validate_path(path)

        syntax_error = self._diff_engine.validate_python_syntax(content, path)
        if syntax_error:
            raise ValueError(f"Syntax validation failed; file not saved. {syntax_error}")

        target.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        
        return f"Created file: {path} ({len(content)} bytes)"

    def apply_edit(self, path: str, search_block: str, replace_block: str) -> str:
        """
        Apply a surgical edit using DiffEngine with fuzzy matching and syntax validation.
        
        Returns:
            Unified diff on success, error message on failure
        """
        target = self._validate_path(path)
        
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(target, "r", encoding="utf-8") as f:
            original_content = f.read()
        
        # Use DiffEngine for fuzzy matching and validation
        result = self._diff_engine.apply_edit(
            filepath=path,
            original_content=original_content,
            search_block=search_block,
            replace_block=replace_block
        )
        
        if not result.success:
            error_msg = result.error
            if result.syntax_error:
                error_msg += f"\n{result.syntax_error}"
            raise ValueError(error_msg)
        
        with open(target, "w", encoding="utf-8") as f:
            f.write(result.new_content)
        
        return result.diff

    # ==================== Terminal Execution ====================

    def run_terminal(self, command: str, cwd: Optional[str] = None) -> TerminalResult:
        """
        Run a terminal command in the workspace.
        
        In Docker mode: executes via `docker exec` with resource limits
        In Local mode: executes via subprocess directly
        """
        start_time = time.time()
        
        if cwd:
            work_dir = self._validate_path(cwd)
        else:
            work_dir = self.workspace_dir
        
        if self.sandbox_type == SandboxType.DOCKER:
            return self._run_docker_exec(command, work_dir, start_time)
        else:
            return self._run_local_exec(command, work_dir, start_time)

    def _run_local_exec(self, command: str, work_dir: Path, start_time: float) -> TerminalResult:
        """Execute command locally via subprocess."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.command_timeout
            )
            duration_ms = int((time.time() - start_time) * 1000)
            return TerminalResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
                duration_ms=duration_ms
            )
        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return TerminalResult(
                exit_code=-1,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
                timed_out=True,
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return TerminalResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=False,
                duration_ms=duration_ms
            )

    def _run_docker_exec(self, command: str, work_dir: Path, start_time: float) -> TerminalResult:
        """Execute command via docker exec."""
        if not self._ensure_docker_container():
            duration_ms = int((time.time() - start_time) * 1000)
            return TerminalResult(
                exit_code=-1,
                stdout="",
                stderr="Docker container not available",
                timed_out=False,
                duration_ms=duration_ms
            )
        
        # Convert workspace path to container path
        try:
            rel_path = work_dir.relative_to(self.workspace_dir)
            container_workdir = f"/workspace/{rel_path}" if str(rel_path) != "." else "/workspace"
        except ValueError:
            container_workdir = "/workspace"
        
        # Build docker exec command
        # Use sh -c to support shell features like pipes, redirects
        docker_cmd = [
            "docker", "exec",
            "-w", container_workdir,
            self._docker_container_id,
            "timeout", "-s", "KILL", str(self.command_timeout),
            "sh", "-c", command
        ]
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.command_timeout + 5
            )
            if result.returncode == 137:
                return TerminalResult(
                    exit_code=-1,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    timed_out=True,
                    duration_ms=int((time.time() - start_time) * 1000)
                )
            duration_ms = int((time.time() - start_time) * 1000)
            return TerminalResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
                duration_ms=duration_ms
            )
        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return TerminalResult(
                exit_code=-1,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
                timed_out=True,
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return TerminalResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=False,
                duration_ms=duration_ms
            )

    def cleanup(self) -> None:
        """Clean up resources (stop Docker container if running)."""
        if self.sandbox_type == SandboxType.DOCKER:
            self._stop_container()


# ==================== Tool Schemas for Function Calling ====================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in the workspace as a tree view",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Relative path from workspace root (default: '.')",
                        "default": "."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file with optional line range",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed start line (inclusive)",
                        "default": 1
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed end line (inclusive), null for end of file"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_edit",
            "description": "Apply a surgical edit to a file using fuzzy search-and-replace with syntax validation. Returns a unified diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root"
                    },
                    "search_block": {
                        "type": "string",
                        "description": "Text to search for (exact or fuzzy match with whitespace tolerance)"
                    },
                    "replace_block": {
                        "type": "string",
                        "description": "Replacement text"
                    }
                },
                "required": ["path", "search_block", "replace_block"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file or overwrite an existing file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Execute a terminal command in the workspace (local or Docker)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute (e.g., 'pytest', 'ruff check', 'git status')"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory relative to workspace root"
                    }
                },
                "required": ["command"]
            }
        }
    }
]


async def execute_tool_call(
    sandbox: SandboxManager,
    tool_name: str,
    arguments: Dict[str, Any]
) -> str:
    """
    Execute a tool call and return the result as a string.
    
    Args:
        sandbox: SandboxManager instance
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        
    Returns:
        Tool result as string
    """
    try:
        if tool_name == "list_files":
            return sandbox.list_files(arguments.get("directory", "."))
        
        elif tool_name == "read_file":
            return sandbox.read_file(
                arguments["path"],
                arguments.get("start_line", 1),
                arguments.get("end_line")
            )
        
        elif tool_name == "apply_edit":
            return sandbox.apply_edit(
                arguments["path"],
                arguments["search_block"],
                arguments["replace_block"]
            )
        
        elif tool_name == "create_file":
            return sandbox.create_file(
                arguments["path"],
                arguments["content"]
            )
        
        elif tool_name == "run_terminal":
            result = sandbox.run_terminal(
                arguments["command"],
                arguments.get("cwd")
            )
            output = f"Exit code: {result.exit_code}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            if result.timed_out:
                output += f"\n[Command timed out after {sandbox.command_timeout}s]\n"
            output += f"Duration: {result.duration_ms}ms"
            return output
        
        else:
            return f"Unknown tool: {tool_name}"
    
    except Exception as e:
        return f"Tool execution error: {str(e)}"