"""
Configuration management for NVIDIA NIM Agentic Coding Assistant.

Provides persistent settings via Pydantic, dynamic model discovery from NIM,
and async HTTP client with exponential backoff for rate limits.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import httpx


class NIMSettings(BaseModel):
    """NVIDIA NIM API settings."""
    api_key: SecretStr = Field(default="", description="NVIDIA API Key")
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


class AppSettings(BaseSettings):
    """Main application settings persisted to settings.json."""
    model_config = SettingsConfigDict(
        json_file="settings.json",
        json_file_encoding="utf-8",
        extra="ignore"
    )

    nim: NIMSettings = Field(default_factory=NIMSettings)
    models: ModelRoleAssignment = Field(default_factory=ModelRoleAssignment)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
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
    """Manages application settings with persistence."""

    def __init__(self, settings_path: str = "settings.json"):
        self.settings_path = Path(settings_path)
        self._settings: Optional[AppSettings] = None

    def load(self) -> AppSettings:
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
        return self._settings

    def save(self, settings: Optional[AppSettings] = None) -> None:
        """Save settings to file."""
        if settings:
            self._settings = settings
        if self._settings:
            data = self._settings.model_dump(mode="json")
            # Don't save the API key in plain text if it's empty
            if not data.get("nim", {}).get("api_key"):
                data["nim"]["api_key"] = ""
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    @property
    def settings(self) -> AppSettings:
        if self._settings is None:
            return self.load()
        return self._settings

    def update_nim_settings(self, api_key: str, base_url: str) -> None:
        """Update NIM API settings."""
        self._settings = self.settings
        self._settings.nim.api_key = api_key
        self._settings.nim.base_url = base_url
        self.save()

    def update_model_assignments(
        self,
        planner: Optional[str] = None,
        coder: Optional[str] = None,
        verifier: Optional[str] = None
    ) -> None:
        """Update model role assignments."""
        self._settings = self.settings
        if planner:
            self._settings.models.planner_model = planner
        if coder:
            self._settings.models.coder_model = coder
        if verifier:
            self._settings.models.verifier_model = verifier
        self.save()

    def update_sandbox_settings(
        self,
        workspace_dir: Optional[str] = None,
        auto_approve: Optional[bool] = None,
        command_timeout: Optional[int] = None
    ) -> None:
        """Update sandbox settings."""
        self._settings = self.settings
        if workspace_dir:
            self._settings.sandbox.workspace_dir = workspace_dir
        if auto_approve is not None:
            self._settings.sandbox.auto_approve_commands = auto_approve
        if command_timeout:
            self._settings.sandbox.command_timeout = command_timeout
        self.save()

    def update_available_models(self, models: List[str]) -> None:
        """Update cached available models list."""
        self._settings = self.settings
        self._settings.available_models = models
        self.save()

    def get_available_model_ids(self) -> List[str]:
        """Get list of available model IDs."""
        return self.settings.available_models


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get or create the global settings manager."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def get_settings() -> AppSettings:
    """Get current application settings."""
    return get_settings_manager().settings


async def discover_models(api_key: str, base_url: str) -> List[str]:
    """Discover available models from NIM."""
    settings = NIMSettings(api_key=api_key, base_url=base_url)
    async with NIMClient(settings) as client:
        models = await client.fetch_available_models()
        return [m.id for m in models]