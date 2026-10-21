"""
Deterministic Hook System for NIM Agentic Coding Assistant.

Provides pre-tool and post-tool hooks for security validation, linting,
and output modification. All hooks are async-compatible for Streamlit integration.
"""

import re
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import fnmatch


class HookType(str, Enum):
    """Types of hooks in the system."""
    PRE_BASH = "pre_bash"
    PRE_EDIT = "pre_edit"
    POST_EDIT = "post_edit"
    POST_BASH = "post_bash"


@dataclass
class HookResult:
    """
    Result of a hook execution.
    
    Attributes:
        allowed: Whether the tool execution should proceed (for pre-hooks)
        message: Human-readable message describing the result
        modified_output: Optional modified output (for post-hooks)
    """
    allowed: bool = True
    message: str = ""
    modified_output: Optional[str] = None


@dataclass
class HookContext:
    """Context passed to hooks during execution."""
    tool_name: str
    arguments: Dict[str, Any]
    workspace_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    """
    Central registry for pre-tool and post-tool hooks.
    
    Hooks are registered by type and executed in registration order.
    Pre-hooks can block execution by returning HookResult(allowed=False).
    Post-hooks can modify output via HookResult.modified_output.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._pre_hooks: Dict[HookType, List[Callable[[HookContext], Awaitable[HookResult]]]] = {
            HookType.PRE_BASH: [],
            HookType.PRE_EDIT: [],
        }
        self._post_hooks: Dict[HookType, List[Callable[[HookContext, Any], Awaitable[HookResult]]]] = {
            HookType.POST_EDIT: [],
            HookType.POST_BASH: [],
        }
        self._config = config or {}
        self._load_patterns_from_config()
    
    def _load_patterns_from_config(self) -> None:
        """Load hook patterns from configuration."""
        self._bash_block_patterns = self._config.get("bash_block_patterns", [])
        self._edit_block_paths = self._config.get("edit_block_paths", [])
        self._sensitive_file_patterns = self._config.get("sensitive_file_patterns", [])
        self._lint_commands = self._config.get("lint_commands", {
            "python": ["ruff", "check", "--output-format=concise"],
            "fallback": ["flake8", "--select=E,W,F"]
        })
    
    def register_pre_hook(
        self,
        hook_type: HookType,
        hook: Callable[[HookContext], Awaitable[HookResult]]
    ) -> None:
        """Register a pre-execution hook."""
        if hook_type in self._pre_hooks:
            self._pre_hooks[hook_type].append(hook)
    
    def register_post_hook(
        self,
        hook_type: HookType,
        hook: Callable[[HookContext, Any], Awaitable[HookResult]]
    ) -> None:
        """Register a post-execution hook."""
        if hook_type in self._post_hooks:
            self._post_hooks[hook_type].append(hook)
    
    async def run_pre_hooks(self, context: HookContext) -> HookResult:
        """
        Run all pre-hooks for the given tool type.
        
        Returns the first blocking result, or a combined success result.
        """
        hooks = self._pre_hooks.get(HookType(context.tool_name), [])
        for hook in hooks:
            result = await hook(context)
            if not result.allowed:
                return result
        return HookResult(allowed=True, message="All pre-hooks passed")
    
    async def run_post_hooks(self, context: HookContext, output: Any) -> HookResult:
        """
        Run all post-hooks for the given tool type.
        
        Returns combined result with potentially modified output.
        """
        hooks = self._post_hooks.get(HookType(context.tool_name), [])
        modified_output = output
        messages = []
        
        for hook in hooks:
            result = await hook(context, modified_output)
            if result.message:
                messages.append(result.message)
            if result.modified_output is not None:
                modified_output = result.modified_output
        
        return HookResult(
            allowed=True,
            message="; ".join(messages) if messages else "All post-hooks completed",
            modified_output=modified_output
        )


# ==================== Default Hook Implementations ====================

# Destructive shell command patterns
DEFAULT_BASH_BLOCK_PATTERNS = [
    # Recursive deletion of root/home/current
    r"rm\s+-rf\s+/($|\s)",
    r"rm\s+-rf\s+/\*\s*($|\s)",
    r"rm\s+-rf\s+~\s*($|\s)",
    r"rm\s+-rf\s+\.\s*($|\s)",
    r"rm\s+-rf\s+\$\w+\s*($|\s)",  # rm -rf $VAR
    
    # Git force push
    r"git\s+push\s+--force\s*($|\s)",
    r"git\s+push\s+-f\s*($|\s)",
    r"git\s+push\s+--force-with-lease\s*($|\s)",
    
    # Credential exfiltration
    r"cat\s+~/\.ssh/.*",
    r"cat\s+/\.ssh/.*",
    r"env\s+\|\s*grep\s+-i\s+(key|secret|token|password|auth)",
    r"printenv\s+\|\s*grep\s+-i\s+(key|secret|token|password|auth)",
    r"cat\s+/etc/passwd",
    r"cat\s+/etc/shadow",
    r"cat\s+/etc/sudoers",
    r"cat\s+~/\.aws/credentials",
    r"cat\s+~/\.config/gcloud/.*",
    r"cat\s+~/\.docker/config\.json",
    r"cat\s+~/\.npmrc",
    r"cat\s+~/\.pypirc",
    r"cat\s+~/\.netrc",
    r"cat\s+~/\.git-credentials",
    r"cat\s+~/\.env",
    r"cat\s+~/\.env\.\w+",
    
    # Dangerous disk operations
    r">\s*/dev/sd[a-z]",
    r">\s*/dev/nvme\d+n\d+",
    r"dd\s+if=",
    r"mkfs\.\w+",
    r"fdisk\s+/dev/",
    r"parted\s+/dev/",
    r"format\s+[a-z]:",
    r"wipefs\s+-a",
    
    # Process killing
    r"kill\s+-9\s+\d+",
    r"pkill\s+-9",
    r"killall\s+-9",
    r"kill\s+-s\s+KILL",
    
    # Network exfiltration / reverse shells
    r"curl\s+.*\|\s*sh",
    r"wget\s+.*\|\s*sh",
    r"curl\s+.*\|\s*bash",
    r"wget\s+.*\|\s*bash",
    r"nc\s+-l\s+\d+",
    r"netcat\s+-l\s+\d+",
    r"nc\s+-e\s+/bin/sh",
    r"netcat\s+-e\s+/bin/sh",
    r"bash\s+-i\s+>&\s*/dev/tcp/",
    r"sh\s+-i\s+>&\s*/dev/tcp/",
    r"/dev/tcp/\d+\.\d+\.\d+\.\d+/\d+",
    
    # Privilege escalation attempts
    r"sudo\s+.*",
    r"su\s+-",
    r"chmod\s+777\s+/",
    r"chown\s+root:root\s+/",
    
    # Kernel/module manipulation
    r"insmod\s+",
    r"rmmod\s+",
    r"modprobe\s+",
    
    # System service manipulation
    r"systemctl\s+(stop|disable|mask)\s+",
    r"service\s+\w+\s+stop",
]

# Sensitive file patterns to block edits
DEFAULT_SENSITIVE_FILE_PATTERNS = [
    # SSH keys and configs
    "**/.ssh/*",
    "**/.ssh/config",
    "**/.ssh/authorized_keys",
    "**/.ssh/known_hosts",
    "**/.ssh/id_*",
    "**/.ssh/*.pem",
    "**/.ssh/*.key",
    
    # AWS/GCP/Azure credentials
    "**/.aws/credentials",
    "**/.aws/config",
    "**/.config/gcloud/*",
    "**/.azure/*",
    
    # Docker credentials
    "**/.docker/config.json",
    "**/.dockercfg",
    
    # Package manager credentials
    "**/.npmrc",
    "**/.pypirc",
    "**/.gem/credentials",
    "**/.cargo/credentials",
    
    # Git credentials
    "**/.git-credentials",
    "**/.gitconfig",
    
    # Environment files
    "**/.env",
    "**/.env.*",
    "**/.env.local",
    "**/.env.production",
    "**/.env.staging",
    
    # Netrc
    "**/.netrc",
    
    # System files
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/hosts",
    "/etc/fstab",
    "/etc/crontab",
    "/etc/ssh/sshd_config",
    
    # Git internals
    "**/.git/*",
    "**/.git/config",
    "**/.git/index",
    "**/.git/HEAD",
    "**/.git/objects/*",
    "**/.git/refs/*",
    
    # IDE/Editor configs with potential secrets
    "**/.vscode/settings.json",
    "**/.idea/workspace.xml",
    
    # Kubernetes
    "**/.kube/config",
    "**/.kube/*",
]

# System directories to block
DEFAULT_SYSTEM_DIRS = [
    "/etc",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/usr/local/bin",
    "/usr/local/sbin",
    "/lib",
    "/lib64",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/run",
    "/var/log",
    "/var/spool",
    "/root",
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\System Volume Information",
]


async def pre_bash_hook(context: HookContext) -> HookResult:
    """
    Block destructive shell commands BEFORE execution.
    
    Checks for:
    - Recursive deletion of critical paths
    - Git force push
    - Credential exfiltration
    - Dangerous disk operations
    - Process killing
    - Network exfiltration / reverse shells
    """
    command = context.arguments.get("command", "")
    if not command:
        return HookResult(allowed=True, message="No command to validate")
    
    # Normalize command for matching
    normalized = command.strip()
    
    # Check against configured patterns
    patterns = context.config.get("bash_block_patterns", DEFAULT_BASH_BLOCK_PATTERNS)
    
    for pattern in patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return HookResult(
                allowed=False,
                message=f"Blocked destructive command: matched pattern '{pattern}'"
            )
    
    # Additional heuristic checks
    # Check for chained dangerous commands
    dangerous_chains = [
        r"&&\s*rm\s+-rf",
        r";\s*rm\s+-rf",
        r"\|\s*sh\s*$",
        r"\|\s*bash\s*$",
    ]
    for pattern in dangerous_chains:
        if re.search(pattern, normalized, re.IGNORECASE):
            return HookResult(
                allowed=False,
                message=f"Blocked dangerous command chain: matched pattern '{pattern}'"
            )
    
    return HookResult(allowed=True, message="Command passed security validation")


async def pre_edit_hook(context: HookContext) -> HookResult:
    """
    Validate file paths before edit operations.
    
    Blocks:
    - System directories
    - .git directory
    - Sensitive files (SSH keys, credentials, env files)
    - Paths outside workspace
    """
    file_path = context.arguments.get("path", "")
    if not file_path:
        return HookResult(allowed=True, message="No file path to validate")
    
    workspace = context.workspace_dir
    target_path = (workspace / file_path).resolve()
    
    # Check if path is outside workspace
    try:
        target_path.relative_to(workspace)
    except ValueError:
        return HookResult(
            allowed=False,
            message=f"Path '{file_path}' is outside workspace directory"
        )
    
    # Check against system directories
    system_dirs = context.config.get("system_dirs", DEFAULT_SYSTEM_DIRS)
    for sys_dir in system_dirs:
        try:
            target_path.relative_to(Path(sys_dir).resolve())
            return HookResult(
                allowed=False,
                message=f"Blocked edit to system directory: {sys_dir}"
            )
        except ValueError:
            pass  # Not under this system dir
    
    # Check against sensitive file patterns
    sensitive_patterns = context.config.get("sensitive_file_patterns", DEFAULT_SENSITIVE_FILE_PATTERNS)
    rel_path = str(target_path.relative_to(workspace))
    
    for pattern in sensitive_patterns:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_path, pattern):
            return HookResult(
                allowed=False,
                message=f"Blocked edit to sensitive file: {file_path} (matched {pattern})"
            )
    
    # Check for .git directory
    if ".git" in target_path.parts:
        return HookResult(
            allowed=False,
            message=f"Blocked edit to .git directory: {file_path}"
        )
    
    return HookResult(allowed=True, message="File path validation passed")


async def post_edit_hook(context: HookContext, output: Any) -> HookResult:
    """
    Run linting on modified Python files after edit.
    
    Runs ruff check or flake8 on the modified file and appends
    warnings to the tool output.
    """
    file_path = context.arguments.get("path", "")
    if not file_path or not file_path.endswith(".py"):
        return HookResult(allowed=True, message="", modified_output=output)
    
    workspace = context.workspace_dir
    target_path = workspace / file_path
    
    if not target_path.exists():
        return HookResult(allowed=True, message="", modified_output=output)
    
    lint_commands = context.config.get("lint_commands", {
        "python": ["ruff", "check", "--output-format=concise"],
        "fallback": ["flake8", "--select=E,W,F"]
    })
    
    # Try ruff first, fallback to flake8
    for cmd_key in ["python", "fallback"]:
        cmd = lint_commands.get(cmd_key, [])
        if not cmd:
            continue
        
        try:
            # Run linter on the specific file
            full_cmd = cmd + [str(target_path)]
            result = await asyncio.to_thread(
                subprocess.run,
                full_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workspace
            )
            
            if result.returncode != 0 and result.stdout.strip():
                # Linter found issues
                lint_output = result.stdout.strip()
                modified = f"{output}\n\n[Lint warnings for {file_path}]:\n{lint_output}"
                return HookResult(
                    allowed=True,
                    message=f"Lint warnings found in {file_path}",
                    modified_output=modified
                )
            elif result.returncode == 0:
                # Clean lint
                return HookResult(
                    allowed=True,
                    message=f"Lint clean for {file_path}",
                    modified_output=output
                )
        except FileNotFoundError:
            # Linter not installed, try next
            continue
        except subprocess.TimeoutExpired:
            return HookResult(
                allowed=True,
                message=f"Lint timeout for {file_path}",
                modified_output=output
            )
        except Exception as e:
            return HookResult(
                allowed=True,
                message=f"Lint error for {file_path}: {str(e)}",
                modified_output=output
            )
    
    return HookResult(allowed=True, message="No linter available", modified_output=output)


async def post_bash_hook(context: HookContext, output: Any) -> HookResult:
    """
    Capture exit codes and detect test failures after bash execution.
    
    Enhances output with:
    - Exit code annotation
    - Test failure detection (pytest, unittest, etc.)
    - Timeout detection
    """
    if not isinstance(output, dict):
        return HookResult(allowed=True, message="", modified_output=output)
    
    # Extract TerminalResult fields
    exit_code = output.get("exit_code", 0)
    stdout = output.get("stdout", "")
    stderr = output.get("stderr", "")
    timed_out = output.get("timed_out", False)
    duration_ms = output.get("duration_ms", 0)
    
    messages = []
    modified_output = dict(output)
    
    # Annotate exit code
    if exit_code != 0:
        messages.append(f"Exit code: {exit_code}")
    
    if timed_out:
        messages.append(f"Command timed out after {duration_ms}ms")
    
    # Detect test failures
    test_patterns = [
        (r"FAILED\s+[\w/]+\.py::", "pytest failures detected"),
        (r"FAIL:\s+test_\w+", "unittest failures detected"),
        (r"\d+ failed,", "test failures detected"),
        (r"ERROR:\s+test_\w+", "test errors detected"),
        (r"AssertionError:", "assertion failures detected"),
    ]
    
    combined_output = stdout + "\n" + stderr
    for pattern, msg in test_patterns:
        if re.search(pattern, combined_output):
            messages.append(msg)
            break
    
    # Detect common error patterns
    error_patterns = [
        (r"SyntaxError:", "Python syntax error"),
        (r"ImportError:", "Import error"),
        (r"ModuleNotFoundError:", "Missing module"),
        (r"Permission denied", "Permission denied"),
        (r"Connection refused", "Connection refused"),
        (r"Timeout", "Operation timed out"),
    ]
    
    for pattern, msg in error_patterns:
        if re.search(pattern, combined_output, re.IGNORECASE):
            messages.append(msg)
    
    if messages:
        modified_output["hook_annotations"] = messages
        modified_output["hook_summary"] = "; ".join(messages)
    
    return HookResult(
        allowed=True,
        message="; ".join(messages) if messages else "Command completed",
        modified_output=modified_output
    )


# ==================== Factory Functions ====================

def create_hook_registry(config: Optional[Dict[str, Any]] = None) -> HookRegistry:
    """
    Create a HookRegistry with default hooks registered.
    
    Args:
        config: Optional configuration dictionary with:
            - bash_block_patterns: List of regex patterns to block
            - edit_block_paths: List of paths to block edits
            - sensitive_file_patterns: List of glob patterns for sensitive files
            - system_dirs: List of system directories to protect
            - lint_commands: Dict with 'python' and 'fallback' command lists
    
    Returns:
        Configured HookRegistry with default hooks
    """
    registry = HookRegistry(config)
    
    # Register default pre-hooks
    registry.register_pre_hook(HookType.PRE_BASH, pre_bash_hook)
    registry.register_pre_hook(HookType.PRE_EDIT, pre_edit_hook)
    
    # Register default post-hooks
    registry.register_post_hook(HookType.POST_EDIT, post_edit_hook)
    registry.register_post_hook(HookType.POST_BASH, post_bash_hook)
    
    return registry


def get_default_hook_config() -> Dict[str, Any]:
    """Get default hook configuration."""
    return {
        "bash_block_patterns": DEFAULT_BASH_BLOCK_PATTERNS,
        "sensitive_file_patterns": DEFAULT_SENSITIVE_FILE_PATTERNS,
        "system_dirs": DEFAULT_SYSTEM_DIRS,
        "lint_commands": {
            "python": ["ruff", "check", "--output-format=concise"],
            "fallback": ["flake8", "--select=E,W,F"]
        }
    }


# ==================== Integration Helper ====================

class HookedSandboxManager:
    """
    Wrapper around SandboxManager that integrates the hook system.
    
    Usage:
        sandbox = SandboxManager(...)
        hooked = HookedSandboxManager(sandbox, config)
        
        # Pre-hooks run before execution
        result = await hooked.run_terminal("ls -la")
        # Post-hooks run after execution
    """
    
    def __init__(
        self,
        sandbox_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        self._sandbox = sandbox_manager
        self._registry = create_hook_registry(config)
        self._workspace_dir = sandbox_manager.workspace_dir
    
    @property
    def registry(self) -> HookRegistry:
        return self._registry
    
    async def run_terminal(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Run terminal command with pre/post hooks."""
        context = HookContext(
            tool_name="pre_bash",
            arguments={"command": command, "cwd": cwd},
            workspace_dir=self._workspace_dir,
            config=self._registry._config
        )
        
        # Run pre-hooks
        pre_result = await self._registry.run_pre_hooks(context)
        if not pre_result.allowed:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Blocked by pre-hook: {pre_result.message}",
                "timed_out": False,
                "duration_ms": 0,
                "blocked": True,
                "block_reason": pre_result.message
            }
        
        # Execute command
        result = self._sandbox.run_terminal(command, cwd)
        
        # Convert to dict for post-hook
        result_dict = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms
        }
        
        # Run post-hooks
        post_context = HookContext(
            tool_name="post_bash",
            arguments={"command": command, "cwd": cwd},
            workspace_dir=self._workspace_dir,
            config=self._registry._config
        )
        post_result = await self._registry.run_post_hooks(post_context, result_dict)
        
        return post_result.modified_output or result_dict
    
    async def apply_edit(self, path: str, search_block: str, replace_block: str) -> str:
        """Apply edit with pre/post hooks."""
        context = HookContext(
            tool_name="pre_edit",
            arguments={"path": path, "search_block": search_block, "replace_block": replace_block},
            workspace_dir=self._workspace_dir,
            config=self._registry._config
        )
        
        # Run pre-hooks
        pre_result = await self._registry.run_pre_hooks(context)
        if not pre_result.allowed:
            raise ValueError(f"Blocked by pre-hook: {pre_result.message}")
        
        # Execute edit
        diff = self._sandbox.apply_edit(path, search_block, replace_block)
        
        # Run post-hooks
        post_context = HookContext(
            tool_name="post_edit",
            arguments={"path": path},
            workspace_dir=self._workspace_dir,
            config=self._registry._config
        )
        post_result = await self._registry.run_post_hooks(post_context, diff)
        
        return post_result.modified_output or diff
    
    # Delegate other methods to sandbox
    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)