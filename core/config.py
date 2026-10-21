"""
Configuration management for NVIDIA NIM Agentic Coding Assistant.

Provides persistent settings via Pydantic, dynamic model discovery from NIM,
and async HTTP client with exponential backoff for rate limits.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, SecretStr
import httpx


class NIMSettings(BaseModel):
    """NVIDIA NIM API settings."""
    model_config = ConfigDict(validate_assignment=True)

    api_key: SecretStr = Field(default=SecretStr(""), description="NVIDIA API Key")
    base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NIM API base URL"
    )
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retry attempts for rate limits")
    base_delay: float = Field(default=1.0, description="Base delay for exponential backoff")


class ModelRoleAssignment(BaseModel):
    """Model role assignments for the multi-agent system."""
    planner_model: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b",
        description="Model for planning tasks"
    )
    coder_model: str = Field(
        default="deepseek-ai/deepseek-v4-flash-0731",
        description="Model for code generation/editing"
    )
    verifier_model: str = Field(
        default="nvidia/nemotron-3.5-lightning-30b-a3b",
        description="Model for verification and testing"
    )


class SandboxSettings(BaseModel):
    """Workspace sandbox settings."""
    workspace_dir: str = Field(default=".", description="Root workspace directory")
    auto_approve_commands: bool = Field(
        default=False,
        description="Auto-run terminal commands without approval"
    )
    command_timeout: int = Field(default=120, description="Terminal command timeout in seconds")
    
    # NEW: Docker sandbox settings
    sandbox_type: Literal["local", "docker"] = Field(
        default="local",
        description="Execution sandbox type: 'local' for subprocess, 'docker' for containerized"
    )
    docker_image: str = Field(
        default="python:3.12-slim",
        description="Docker image to use for sandbox container"
    )
    docker_cpu_limit: str = Field(
        default="1.0",
        description="CPU limit for Docker container (e.g., '1.0' for 1 CPU)"
    )
    docker_memory_limit: str = Field(
        default="2g",
        description="Memory limit for Docker container (e.g., '2g' for 2GB)"
    )
    auto_git_checkpoint: bool = Field(
        default=True,
        description="Automatically create git checkpoint before agent runs"
    )


class RepoMapSettings(BaseModel):
    """Repository map settings for codebase navigation."""
    token_budget: int = Field(
        default=2000,
        description="Maximum tokens for repo map output"
    )
    include_patterns: List[str] = Field(
        default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"],
        description="File patterns to include in repo map"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["**/node_modules/**", "**/.git/**", "**/__pycache__/**", "**/venv/**", "**/.venv/**", "**/dist/**", "**/build/**"],
        description="File patterns to exclude from repo map"
    )
    max_file_size_kb: int = Field(
        default=100,
        description="Maximum file size (KB) to parse for repo map"
    )


class ContextSettings(BaseModel):
    """Context management settings for token-efficient conversations."""
    window_tokens: int = Field(
        default=8000,
        description="Target token budget for context window"
    )
    keep_last_turns: int = Field(
        default=3,
        description="Number of recent tool turns to keep in full detail"
    )
    summary_max_lines: int = Field(
        default=2,
        description="Maximum lines for compressed summaries"
    )
    token_estimation_method: Literal["tiktoken", "heuristic"] = Field(
        default="tiktoken",
        description="Token estimation method"
    )


class MCPServerConfig(BaseModel):
    """MCP server configuration."""
    model_config = ConfigDict(validate_assignment=True)

    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        default="stdio",
        description="Transport protocol for MCP server"
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to start the server (for stdio transport)"
    )
    args: List[str] = Field(
        default_factory=list,
        description="Arguments for the command"
    )
    url: Optional[str] = Field(
        default=None,
        description="Server URL (for sse or streamable-http transport)"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers for HTTP transports"
    )
    enabled: bool = Field(
        default=False,
        description="Whether this server is enabled"
    )
    timeout: float = Field(
        default=30.0,
        description="Connection timeout in seconds"
    )


class MCPSettings(BaseModel):
    """MCP (Model Context Protocol) settings."""
    model_config = ConfigDict(validate_assignment=True)

    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP server configurations by name"
    )


class TelemetrySettings(BaseModel):
    """Telemetry and trajectory export settings."""
    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(
        default=True,
        description="Enable telemetry collection"
    )
    export_dir: str = Field(
        default="./trajectories",
        description="Directory for trajectory exports"
    )
    redundant_threshold: int = Field(
        default=2,
        description="Threshold for redundant action detection"
    )
    sample_rate: float = Field(
        default=1.0,
        description="Sampling rate for telemetry (0.0 to 1.0)"
    )


class HookSettings(BaseModel):
    """Hook settings for pre/post action validation."""
    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(
        default=True,
        description="Enable hooks system"
    )
    block_destructive_commands: bool = Field(
        default=True,
        description="Block potentially destructive commands"
    )
    run_linter_on_edit: bool = Field(
        default=True,
        description="Run linter after file edits"
    )
    linter: str = Field(
        default="ruff",
        description="Linter to use (ruff, pylint, flake8, etc.)"
    )
    custom_patterns: List[str] = Field(
        default_factory=list,
        description="Custom regex patterns for additional checks"
    )


class AppSettings(BaseModel):
    """Main application settings persisted to settings.json."""
    model_config = ConfigDict(extra="ignore")

    nim: NIMSettings = Field(default_factory=NIMSettings)
    models: ModelRoleAssignment = Field(default_factory=ModelRoleAssignment)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    repo_map: RepoMapSettings = Field(default_factory=RepoMapSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    hooks: HookSettings = Field(default_factory=HookSettings)
    available_models: List[str] = Field(default_factory=list, description="Cached model list from NIM")


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "nvidia"


class NIMClient:
    """Async HTTP client for NVIDIA NIM API with exponential backoff."""

    def __init__(self, settings: NIMSettings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "NIMClient":
        self._client = httpx.AsyncClient(
            base_url=self.settings.base_url,
            headers={
                "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
                "Content-Type": "application/json"
            },
            timeout=self.settings.timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _request_with_backoff(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with exponential backoff for 429 errors."""
        delay = self.settings.base_delay
        last_exception = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)

                if response.status_code == 429:
                    if attempt < self.settings.max_retries:
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                        continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self.settings.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
            except Exception as e:
                last_exception = e
                if attempt < self.settings.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise

        raise last_exception or Exception("Max retries exceeded")

    async def fetch_available_models(self) -> List[ModelInfo]:
        """Fetch available models from NIM /v1/models endpoint."""
        response = await self._request_with_backoff("GET", "/models")
        data = response.json()

        models = []
        for item in data.get("data", []):
            models.append(ModelInfo(
                id=item.get("id", ""),
                object=item.get("object", "model"),
                created=item.get("created", 0),
                owned_by=item.get("owned_by", "nvidia")
            ))
        return models

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 8192
    ) -> httpx.Response:
        """Create a chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        return await self._request_with_backoff("POST", "/chat/completions", json=payload)

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192
    ):
        """Stream chat completion responses."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield data


class SettingsManager:
    """Manages application settings with persistence to settings.json."""

    def __init__(self, settings_path: str = "settings.json"):
        self.settings_path = Path(settings_path)
        self._settings: Optional[AppSettings] = None
        self._load()

    def _load(self):
        """Load settings from file or create defaults."""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._settings = AppSettings(**data)
            except Exception:
                self._settings = AppSettings()
        else:
            self._settings = AppSettings()
            self.save()

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def save(self):
        """Persist settings to JSON file."""
        data = self._settings.model_dump(mode="json")
        data["nim"]["api_key"] = self._settings.nim.api_key.get_secret_value()
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def update_nim_settings(self, api_key: str, base_url: str):
        """Update NIM API settings."""
        self._settings.nim.api_key = api_key
        self._settings.nim.base_url = base_url
        self.save()

    def update_model_assignments(self, planner: str, coder: str, verifier: str):
        """Update model role assignments."""
        self._settings.models.planner_model = planner
        self._settings.models.coder_model = coder
        self._settings.models.verifier_model = verifier
        self.save()

    def update_sandbox_settings(
        self,
        workspace_dir: str,
        auto_approve: bool,
        command_timeout: int,
        sandbox_type: str = "local",
        docker_image: str = "python:3.12-slim",
        docker_cpu_limit: str = "1.0",
        docker_memory_limit: str = "2g",
        auto_git_checkpoint: bool = True
    ):
        """Update sandbox settings."""
        self._settings.sandbox.workspace_dir = workspace_dir
        self._settings.sandbox.auto_approve_commands = auto_approve
        self._settings.sandbox.command_timeout = command_timeout
        self._settings.sandbox.sandbox_type = sandbox_type  # type: ignore
        self._settings.sandbox.docker_image = docker_image
        self._settings.sandbox.docker_cpu_limit = docker_cpu_limit
        self._settings.sandbox.docker_memory_limit = docker_memory_limit
        self._settings.sandbox.auto_git_checkpoint = auto_git_checkpoint
        self.save()

    def update_repo_map_settings(
        self,
        token_budget: int = 2000,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        max_file_size_kb: int = 100
    ):
        """Update repo map settings."""
        self._settings.repo_map.token_budget = token_budget
        if include_patterns is not None:
            self._settings.repo_map.include_patterns = include_patterns
        if exclude_patterns is not None:
            self._settings.repo_map.exclude_patterns = exclude_patterns
        self._settings.repo_map.max_file_size_kb = max_file_size_kb
        self.save()

    def update_context_settings(
        self,
        window_tokens: int = 8000,
        keep_last_turns: int = 3,
        summary_max_lines: int = 2,
        token_estimation_method: str = "tiktoken"
    ):
        """Update context management settings."""
        self._settings.context.window_tokens = window_tokens
        self._settings.context.keep_last_turns = keep_last_turns
        self._settings.context.summary_max_lines = summary_max_lines
        self._settings.context.token_estimation_method = token_estimation_method  # type: ignore
        self.save()

    def update_available_models(self, models: List[str]):
        """Update cached available models list."""
        self._settings.available_models = models
        self.save()

        def update_mcp_settings(self, servers: Dict[str, Dict[str, Any]]):
            """Update MCP server settings."""
            mcp_servers = {}
            for name, config in servers.items():
                mcp_servers[name] = MCPServerConfig(**config)
            self._settings.mcp.servers = mcp_servers
            self.save()

        def update_telemetry_settings(
            self,
            enabled: bool = True,
            export_dir: str = "./trajectories",
            redundant_threshold: int = 2,
            sample_rate: float = 1.0
        ):
            """Update telemetry settings."""
            self._settings.telemetry.enabled = enabled
            self._settings.telemetry.export_dir = export_dir
            self._settings.telemetry.redundant_threshold = redundant_threshold
            self._settings.telemetry.sample_rate = sample_rate
            self.save()

        def update_hook_settings(
            self,
            enabled: bool = True,
            block_destructive_commands: bool = True,
            run_linter_on_edit: bool = True,
            linter: str = "ruff",
            custom_patterns: List[str] = None
        ):
            """Update hook settings."""
            self._settings.hooks.enabled = enabled
            self._settings.hooks.block_destructive_commands = block_destructive_commands
            self._settings.hooks.run_linter_on_edit = run_linter_on_edit
            self._settings.hooks.linter = linter
            if custom_patterns is not None:
                self._settings.hooks.custom_patterns = custom_patterns
            self.save()


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager(settings_path: str = "settings.json") -> SettingsManager:
    """Get or create the global settings manager."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager(settings_path)
    return _settings_manager


def get_settings() -> AppSettings:
    """Get current application settings."""
    return get_settings_manager().settings


async def discover_models(api_key: str, base_url: str) -> List[str]:
    """Discover available models from NIM."""
    nim_settings = NIMSettings(api_key=api_key, base_url=base_url)
    async with NIMClient(nim_settings) as client:
        models = await client.fetch_available_models()
        return [m.id for m in models]