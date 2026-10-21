"""
MCP (Model Context Protocol) Client Bridge for NIM Agentic Coding Assistant.

Provides dynamic tool discovery and registration from external MCP servers
with support for stdio (subprocess) and SSE (HTTP/SSE) transports.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union, Tuple
from enum import Enum

import httpx

from .config import get_settings_manager, AppSettings

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    transport: Literal["stdio", "sse"]
    
    # stdio transport fields
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    
    # sse transport fields
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    # common fields
    enabled: bool = True
    timeout: float = 30.0
    
    def __post_init__(self):
        """Validate configuration based on transport type."""
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"MCP server '{self.name}': command required for stdio transport")
        elif self.transport == "sse":
            if not self.url:
                raise ValueError(f"MCP server '{self.name}': url required for sse transport")
        else:
            raise ValueError(f"MCP server '{self.name}': unknown transport '{self.transport}'")


@dataclass
class MCPTool:
    """Represents a discovered MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    namespaced_name: str
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.namespaced_name,
                "description": f"[{self.server_name}] {self.description}",
                "parameters": self.input_schema
            }
        }


class MCPServerConnection:
    """Manages a single MCP server connection."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._sse_client: Optional[httpx.AsyncClient] = None
        self._tools: List[MCPTool] = []
        self._healthy = False
        self._last_health_check = 0.0
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._base_reconnect_delay = 1.0
        
    @property
    def is_healthy(self) -> bool:
        return self._healthy
    
    @property
    def tools(self) -> List[MCPTool]:
        return self._tools
    
    async def start(self) -> bool:
        """Start the MCP server connection."""
        if self.config.transport == "stdio":
            return await self._start_stdio()
        else:
            return await self._start_sse()
    
    async def _start_stdio(self) -> bool:
        """Start stdio transport via subprocess."""
        try:
            env = os.environ.copy()
            env.update(self.config.env)
            
            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1
            )
            
            # Initialize MCP connection
            await self._send_initialize()
            await self._discover_tools()
            self._healthy = True
            self._reconnect_attempts = 0
            logger.info(f"MCP server '{self.config.name}' started via stdio")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server '{self.config.name}' via stdio: {e}")
            self._healthy = False
            return False
    
    async def _start_sse(self) -> bool:
        """Start SSE transport via HTTP."""
        try:
            self._sse_client = httpx.AsyncClient(
                base_url=self.config.url,
                headers=self.config.headers,
                timeout=self.config.timeout
            )
            
            # Test connection with a simple request
            await self._send_initialize()
            await self._discover_tools()
            self._healthy = True
            self._reconnect_attempts = 0
            logger.info(f"MCP server '{self.config.name}' connected via SSE")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.config.name}' via SSE: {e}")
            self._healthy = False
            if self._sse_client:
                await self._sse_client.aclose()
                self._sse_client = None
            return False
    
    async def _send_initialize(self) -> None:
        """Send MCP initialize request."""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "nim-agentic-assistant",
                    "version": "1.0.0"
                }
            }
        }
        
        if self.config.transport == "stdio":
            await self._send_stdio_request(init_request)
        else:
            await self._send_sse_request(init_request)
    
    async def _send_stdio_request(self, request: Dict) -> Dict:
        """Send request via stdio and read response."""
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("Process not running")
        
        request_json = json.dumps(request) + "\n"
        self._process.stdin.write(request_json)
        self._process.stdin.flush()
        
        # Read response with timeout
        response_line = await asyncio.wait_for(
            asyncio.to_thread(self._process.stdout.readline),
            timeout=self.config.timeout
        )
        
        if not response_line:
            raise RuntimeError("No response from MCP server")
        
        return json.loads(response_line.strip())
    
    async def _send_sse_request(self, request: Dict) -> Dict:
        """Send request via SSE/HTTP."""
        if not self._sse_client:
            raise RuntimeError("SSE client not initialized")
        
        response = await self._sse_client.post("/mcp", json=request)
        response.raise_for_status()
        return response.json()
    
    async def _discover_tools(self) -> None:
        """Discover tools from the MCP server."""
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        if self.config.transport == "stdio":
            response = await self._send_stdio_request(tools_request)
        else:
            response = await self._send_sse_request(tools_request)
        
        if "result" in response and "tools" in response["result"]:
            self._tools = []
            for tool_data in response["result"]["tools"]:
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {"type": "object", "properties": {}}),
                    server_name=self.config.name,
                    namespaced_name=f"mcp.{self.config.name}.{tool_data['name']}"
                )
                self._tools.append(tool)
            logger.info(f"Discovered {len(self._tools)} tools from MCP server '{self.config.name}'")
        else:
            logger.warning(f"No tools found from MCP server '{self.config.name}'")
            self._tools = []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self._healthy:
            raise RuntimeError(f"MCP server '{self.config.name}' is not healthy")
        
        # Find the tool
        tool = next((t for t in self._tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found on server '{self.config.name}'")
        
        call_request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        if self.config.transport == "stdio":
            response = await self._send_stdio_request(call_request)
        else:
            response = await self._send_sse_request(call_request)
        
        if "error" in response:
            raise RuntimeError(f"MCP tool error: {response['error']}")
        
        return response.get("result", {})
    
    async def health_check(self) -> bool:
        """Perform health check on the connection."""
        now = time.time()
        if now - self._last_health_check < 30:  # Cache for 30 seconds
            return self._healthy
        
        self._last_health_check = now
        
        try:
            # Simple ping via tools/list
            tools_request = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": "tools/list",
                "params": {}
            }
            
            if self.config.transport == "stdio":
                if not self._process or self._process.poll() is not None:
                    self._healthy = False
                    return False
                await self._send_stdio_request(tools_request)
            else:
                if not self._sse_client:
                    self._healthy = False
                    return False
                await self._send_sse_request(tools_request)
            
            self._healthy = True
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for MCP server '{self.config.name}': {e}")
            self._healthy = False
            return False
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnect attempts reached for MCP server '{self.config.name}'")
            return False
        
        delay = self._base_reconnect_delay * (2 ** self._reconnect_attempts)
        self._reconnect_attempts += 1
        
        logger.info(f"Reconnecting to MCP server '{self.config.name}' in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)
        
        await self.stop()
        return await self.start()
    
    async def stop(self) -> None:
        """Stop the MCP server connection."""
        self._healthy = False
        
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(
                    asyncio.to_thread(self._process.wait),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self._process.kill()
                await asyncio.to_thread(self._process.wait)
            except Exception:
                pass
            self._process = None
        
        if self._sse_client:
            await self._sse_client.aclose()
            self._sse_client = None
        
        self._tools = []
        logger.info(f"MCP server '{self.config.name}' stopped")


class MCPClientManager:
    """Manages connections to multiple MCP servers."""
    
    def __init__(self, settings: Optional[AppSettings] = None):
        self._settings = settings or get_settings_manager().settings
        self._connections: Dict[str, MCPServerConnection] = {}
        self._running = False
    
    @property
    def connections(self) -> Dict[str, MCPServerConnection]:
        return self._connections
    
    def load_server_configs(self) -> List[MCPServerConfig]:
        """Load MCP server configurations from settings."""
        mcp_config = self._settings.model_dump(mode="json").get("mcp", {})
        servers_config = mcp_config.get("servers", {})
        
        configs = []
        for name, config_data in servers_config.items():
            try:
                config = MCPServerConfig(name=name, **config_data)
                configs.append(config)
            except Exception as e:
                logger.error(f"Invalid MCP server config '{name}': {e}")
        
        return configs
    
    async def start_all(self) -> Dict[str, bool]:
        """Start all enabled MCP server connections."""
        self._running = True
        configs = self.load_server_configs()
        results = {}
        
        for config in configs:
            if not config.enabled:
                logger.info(f"MCP server '{config.name}' is disabled, skipping")
                results[config.name] = False
                continue
            
            connection = MCPServerConnection(config)
            self._connections[config.name] = connection
            results[config.name] = await connection.start()
        
        return results
    
    async def stop_all(self) -> None:
        """Stop all MCP server connections."""
        self._running = False
        
        for name, connection in self._connections.items():
            await connection.stop()
        
        self._connections.clear()
        logger.info("All MCP connections stopped")
    
    async def restart_server(self, name: str) -> bool:
        """Restart a specific MCP server."""
        if name not in self._connections:
            return False
        
        connection = self._connections[name]
        await connection.stop()
        return await connection.start()
    
    async def get_server_tools(self, name: str) -> List[MCPTool]:
        """Get tools from a specific server."""
        if name in self._connections:
            return self._connections[name].tools
        return []
    
    async def get_all_tools(self) -> List[MCPTool]:
        """Get all tools from all connected servers."""
        all_tools = []
        for connection in self._connections.values():
            if connection.is_healthy:
                all_tools.extend(connection.tools)
        return all_tools
    
    async def call_tool(self, namespaced_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by its namespaced name (mcp.{server}.{tool})."""
        # Parse namespaced name
        parts = namespaced_name.split(".", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError(f"Invalid namespaced tool name: {namespaced_name}")
        
        server_name = parts[1]
        tool_name = parts[2]
        
        if server_name not in self._connections:
            raise ValueError(f"MCP server '{server_name}' not connected")
        
        connection = self._connections[server_name]
        if not connection.is_healthy:
            # Try to reconnect
            await connection.reconnect()
            if not connection.is_healthy:
                raise RuntimeError(f"MCP server '{server_name}' is unavailable")
        
        return await connection.call_tool(tool_name, arguments)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Health check all connections."""
        results = {}
        for name, connection in self._connections.items():
            results[name] = await connection.health_check()
        return results
    
    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections for UI."""
        status = {}
        for name, connection in self._connections.items():
            status[name] = {
                "name": name,
                "transport": connection.config.transport,
                "enabled": connection.config.enabled,
                "healthy": connection.is_healthy,
                "tool_count": len(connection.tools),
                "reconnect_attempts": connection._reconnect_attempts
            }
        return status


class ToolRegistry:
    """Dynamic tool registry merging local and MCP tools."""
    
    def __init__(self, mcp_manager: MCPClientManager):
        self._mcp_manager = mcp_manager
        self._local_tools: List[Dict[str, Any]] = []
        self._mcp_tools: List[MCPTool] = []
        self._merged_schemas: List[Dict[str, Any]] = []
    
    def register_local_tools(self, tool_schemas: List[Dict[str, Any]]) -> None:
        """Register local tool schemas (e.g., from sandbox.executor.TOOL_SCHEMAS)."""
        self._local_tools = tool_schemas
        self._rebuild_merged()
    
    async def refresh_mcp_tools(self) -> None:
        """Refresh MCP tools from connected servers."""
        self._mcp_tools = await self._mcp_manager.get_all_tools()
        self._rebuild_merged()
    
    def _rebuild_merged(self) -> None:
        """Rebuild merged tool schema list."""
        self._merged_schemas = []
        
        # Add local tools
        self._merged_schemas.extend(self._local_tools)
        
        # Add MCP tools with namespaced names
        for tool in self._mcp_tools:
            self._merged_schemas.append(tool.to_openai_schema())
        
        logger.info(f"Tool registry rebuilt: {len(self._local_tools)} local + {len(self._mcp_tools)} MCP = {len(self._merged_schemas)} total")
    
    @property
    def all_tools(self) -> List[Dict[str, Any]]:
        """Get all merged tool schemas for LLM function calling."""
        return self._merged_schemas
    
    @property
    def local_tools(self) -> List[Dict[str, Any]]:
        return self._local_tools
    
    @property
    def mcp_tools(self) -> List[MCPTool]:
        return self._mcp_tools
    
    def get_tool_by_name(self, name: str) -> Optional[Union[Dict[str, Any], MCPTool]]:
        """Find a tool by name (local or MCP)."""
        # Check local tools
        for tool in self._local_tools:
            if tool.get("function", {}).get("name") == name:
                return tool
        
        # Check MCP tools
        for tool in self._mcp_tools:
            if tool.namespaced_name == name:
                return tool
        
        return None
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name (local or MCP)."""
        tool = self.get_tool_by_name(name)
        
        if tool is None:
            raise ValueError(f"Tool not found: {name}")
        
        if isinstance(tool, MCPTool):
            # MCP tool - delegate to manager
            return await self._mcp_manager.call_tool(name, arguments)
        else:
            # Local tool - would need sandbox executor
            # This is handled by the agent orchestrator
            raise NotImplementedError(f"Local tool execution not implemented in registry: {name}")


# Global instances
_mcp_manager: Optional[MCPClientManager] = None
_tool_registry: Optional[ToolRegistry] = None


def get_mcp_manager(settings: Optional[AppSettings] = None) -> MCPClientManager:
    """Get or create the global MCP client manager."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager(settings)
    return _mcp_manager


def get_tool_registry(mcp_manager: Optional[MCPClientManager] = None) -> ToolRegistry:
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        manager = mcp_manager or get_mcp_manager()
        _tool_registry = ToolRegistry(manager)
    return _tool_registry


async def initialize_mcp(
    settings: Optional[AppSettings] = None,
    local_tool_schemas: Optional[List[Dict[str, Any]]] = None
) -> Tuple[MCPClientManager, ToolRegistry]:
    """
    Initialize MCP system and return manager and registry.
    
    Args:
        settings: Optional AppSettings (uses global if not provided)
        local_tool_schemas: Local tool schemas to register (e.g., TOOL_SCHEMAS from sandbox.executor)
        
    Returns:
        Tuple of (MCPClientManager, ToolRegistry)
    """
    manager = get_mcp_manager(settings)
    registry = get_tool_registry(manager)
    
    if local_tool_schemas:
        registry.register_local_tools(local_tool_schemas)
    
    await manager.start_all()
    await registry.refresh_mcp_tools()
    
    return manager, registry


async def shutdown_mcp() -> None:
    """Shutdown MCP system gracefully."""
    global _mcp_manager, _tool_registry
    
    if _mcp_manager:
        await _mcp_manager.stop_all()
        _mcp_manager = None
    
    _tool_registry = None
    logger.info("MCP system shutdown complete")


# Convenience function for getting all tools (local + MCP)
async def get_all_tools(
    settings: Optional[AppSettings] = None,
    local_tool_schemas: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Get all merged tool schemas (local + MCP).
    
    This is the main entry point for the agent orchestrator to get
    the complete tool list for function calling.
    """
    manager, registry = await initialize_mcp(settings, local_tool_schemas)
    return registry.all_tools