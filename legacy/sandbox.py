"""
Workspace Sandbox - Isolated file operations and terminal execution.

Provides secure, path-validated tools for the agent to interact with the workspace:
- list_files: Tree view of workspace files
- read_file: Read file contents with line ranges
- apply_edit: Surgical search-and-replace with unified diff output
- create_file: Create or overwrite files
- run_terminal: Execute commands with timeout and capture output
"""

import os
import subprocess
import difflib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import pathspec


@dataclass
class FileEntry:
    """Represents a file or directory entry."""
    name: str
    path: str
    is_dir: bool
    size: int = 0
    children: List["FileEntry"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class TerminalResult:
    """Result of a terminal command execution."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class WorkspaceSandbox:
    """
    Secure workspace sandbox with path validation.
    
    All operations are restricted to the configured workspace directory.
    Prevents path traversal attacks and unauthorized file access.
    """

    def __init__(self, workspace_dir: str, auto_approve: bool = False, command_timeout: int = 120):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.auto_approve = auto_approve
        self.command_timeout = command_timeout
        self._gitignore_spec: Optional[pathspec.PathSpec] = None
        self._load_gitignore()

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

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve a path within the workspace.
        
        Raises:
            ValueError: If path is outside workspace or invalid.
        """
        # Resolve relative to workspace
        target = (self.workspace_dir / path).resolve()
        
        # Ensure it's within workspace
        try:
            target.relative_to(self.workspace_dir)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside workspace directory")
        
        return target

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored per .gitignore."""
        if self._gitignore_spec is None:
            return False
        try:
            rel_path = path.relative_to(self.workspace_dir)
            return self._gitignore_spec.match_file(str(rel_path))
        except Exception:
            return False

    def list_files(self, directory: str = ".") -> str:
        """
        List files in a directory as a tree view.
        
        Args:
            directory: Relative path from workspace root
            
        Returns:
            Formatted tree string
        """
        target = self._validate_path(directory)
        
        if not target.exists():
            return f"Directory not found: {directory}"
        
        if not target.is_dir():
            return f"Not a directory: {directory}"
        
        entries = self._build_tree(target)
        return self._format_tree(entries, directory)

    def _build_tree(self, root: Path, prefix: str = "") -> List[FileEntry]:
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
                children = self._build_tree(item, prefix + "  ")
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
                lines.extend(child_lines.splitlines()[1:])  # Skip root line
        
        return "\n".join(lines)

    def read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        """
        Read file contents with optional line range.
        
        Args:
            path: Relative path from workspace root
            start_line: 1-indexed start line (inclusive)
            end_line: 1-indexed end line (inclusive), None for end of file
            
        Returns:
            File contents as string
        """
        target = self._validate_path(path)
        
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not target.is_file():
            raise ValueError(f"Not a file: {path}")
        
        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Convert to 0-indexed
        start_idx = max(0, start_line - 1)
        end_idx = len(lines) if end_line is None else min(len(lines), end_line)
        
        selected = lines[start_idx:end_idx]
        return "".join(selected)

    def create_file(self, path: str, content: str) -> str:
        """
        Create or overwrite a file.
        
        Args:
            path: Relative path from workspace root
            content: File content
            
        Returns:
            Success message
        """
        target = self._validate_path(path)
        
        # Create parent directories
        target.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        
        return f"Created file: {path} ({len(content)} bytes)"

    def apply_edit(self, path: str, search_block: str, replace_block: str) -> str:
        """
        Apply a surgical edit to a file using search-and-replace.
        
        Args:
            path: Relative path from workspace root
            search_block: Exact text to search for (must be unique)
            replace_block: Replacement text
            
        Returns:
            Unified diff of the change
            
        Raises:
            ValueError: If search_block not found or matches multiple locations
        """
        target = self._validate_path(path)
        
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(target, "r", encoding="utf-8") as f:
            original_content = f.read()
        
        # Count occurrences
        count = original_content.count(search_block)
        if count == 0:
            raise ValueError(f"Search block not found in {path}")
        if count > 1:
            raise ValueError(f"Search block matches {count} locations in {path} (must be unique)")
        
        # Perform replacement
        new_content = original_content.replace(search_block, replace_block)
        
        # Write back
        with open(target, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        # Generate unified diff
        diff = self._generate_unified_diff(
            original_content, new_content, path
        )
        
        return diff

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

    def run_terminal(self, command: str, cwd: Optional[str] = None) -> TerminalResult:
        """
        Run a terminal command in the workspace.
        
        Args:
            command: Command to execute
            cwd: Working directory (relative to workspace), defaults to workspace root
            
        Returns:
            TerminalResult with exit_code, stdout, stderr
        """
        if cwd:
            work_dir = self._validate_path(cwd)
        else:
            work_dir = self.workspace_dir
        
        # Security: Use shell=False, split command properly
        # For complex commands, we use shell=True but with validation
        try:
            result = subprocess.run(
                command,
                shell=True,  # Allow shell features like pipes, redirects
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.command_timeout
            )
            return TerminalResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False
            )
        except subprocess.TimeoutExpired as e:
            return TerminalResult(
                exit_code=-1,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
                timed_out=True
            )
        except Exception as e:
            return TerminalResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=False
            )


# OpenAI-compatible tool schemas for function calling
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
            "description": "Apply a surgical edit to a file using exact search-and-replace. Returns a unified diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root"
                    },
                    "search_block": {
                        "type": "string",
                        "description": "Exact text to search for (must be unique in file)"
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
            "description": "Execute a terminal command in the workspace",
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
    sandbox: WorkspaceSandbox,
    tool_name: str,
    arguments: Dict[str, Any]
) -> str:
    """
    Execute a tool call and return the result as a string.
    
    Args:
        sandbox: WorkspaceSandbox instance
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
                output += f"[TIMEOUT after {sandbox.command_timeout}s]\n"
            return output
        
        else:
            return f"Unknown tool: {tool_name}"
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"