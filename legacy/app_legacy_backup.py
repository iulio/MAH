"""
NVIDIA NIM Agentic Coding Assistant - Streamlit Copilot-style Chat UI.

Features:
- Left Sidebar: Settings, Model Selection, Workspace Explorer
- Main Panel: Multi-turn chat with @file mentions, reasoning blocks, tool calls, diff viewer
- Two modes: Ask (read-only) and Agent (full autonomous loop with self-healing)
"""

import streamlit as st
import asyncio
import atexit
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from core import (
    SettingsManager, get_settings_manager,
    discover_models,
    AgentHarness, AgentMode, build_sandbox,
    create_repo_mapper,
)
from sandbox import SandboxManager, GitGuard

# Page configuration
st.set_page_config(
    page_title="NIM Agentic Coding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Copilot-style UI
st.markdown("""
<style>
    /* Main layout */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        animation: fadeIn 0.2s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message.user {
        background-color: #2d2d2d;
        border-left: 3px solid #007acc;
    }
    
    .chat-message.assistant {
        background-color: #1e1e1e;
        border-left: 3px solid #4ec9b0;
    }
    
    .chat-message.tool {
        background-color: #2d2d2d;
        border-left: 3px solid #dcdcaa;
        font-family: 'Monospace', monospace;
        font-size: 0.85rem;
    }
    
    .chat-message.system {
        background-color: #2d2d2d;
        border-left: 3px solid #ce9178;
        font-style: italic;
    }
    
    /* Reasoning block */
    .reasoning-block {
        background-color: #1a1a2e;
        border: 1px solid #3d3d5c;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #b5cea8;
    }
    
    .reasoning-header {
        color: #9cdcfe;
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Tool call block */
    .tool-call-block {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .tool-call-header {
        color: #dcdcaa;
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .tool-call-args {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 0.5rem;
        font-family: 'Monospace', monospace;
        font-size: 0.8rem;
        overflow-x: auto;
    }
    
    .tool-call-result {
        background-color: #1e1e1e;
        border-radius: 4px;
        padding: 0.5rem;
        font-family: 'Monospace', monospace;
        font-size: 0.8rem;
        overflow-x: auto;
        margin-top: 0.5rem;
        border-left: 3px solid #4ec9b0;
    }
    
    /* Diff viewer */
    .diff-viewer {
        background-color: #1e1e1e;
        border: 1px solid #3d3d3d;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .diff-header {
        background-color: #2d2d2d;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #3d3d3d;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .diff-content {
        padding: 1rem;
        font-family: 'Monospace', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow: auto;
    }
    
    .diff-line {
        padding: 0.1rem 0.5rem;
        white-space: pre-wrap;
        word-break: break-all;
    }
    
    .diff-line.added {
        background-color: #2d4a2d;
        color: #b5cea8;
    }
    
    .diff-line.removed {
        background-color: #4a2d2d;
        color: #f48771;
    }
    
    .diff-line.context {
        color: #cccccc;
    }
    
    .diff-line.header {
        color: #9cdcfe;
    }
    
    /* File tree */
    .file-tree {
        font-family: 'Monospace', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    .file-tree-item {
        padding: 0.1rem 0.5rem;
        cursor: pointer;
        border-radius: 3px;
        transition: background 0.1s;
    }
    
    .file-tree-item:hover {
        background-color: #3d3d3d;
    }
    
    .file-tree-item.selected {
        background-color: #004578;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .status-plan { background-color: #007acc; color: white; }
    .status-edit { background-color: #4ec9b0; color: #1e1e1e; }
    .status-verify { background-color: #dcdcaa; color: #1e1e1e; }
    .status-fix { background-color: #ce9178; color: #1e1e1e; }
    .status-success { background-color: #4ec9b0; color: white; }
    .status-failure { background-color: #f48771; color: white; }
    
    /* Input area */
    .stChatInputContainer {
        background-color: #1e1e1e;
        border-top: 1px solid #3d3d3d;
        padding: 1rem;
    }
    
    /* Model selector */
    .model-selector {
        background-color: #2d2d2d;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        border-radius: 6px;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.1s;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #007acc;
        border-color: #007acc;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #005a9e;
        border-color: #005a9e;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3d3d3d;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4d4d4d;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "harness" not in st.session_state:
        st.session_state.harness = None
    if "sandbox" not in st.session_state:
        st.session_state.sandbox = None
    if "settings_manager" not in st.session_state:
        st.session_state.settings_manager = get_settings_manager()
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = AgentMode.ASK
    if "workspace_files" not in st.session_state:
        st.session_state.workspace_files = []
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "show_diff" not in st.session_state:
        st.session_state.show_diff = None
    if "pending_diff" not in st.session_state:
        st.session_state.pending_diff = None
    if "agent_running" not in st.session_state:
        st.session_state.agent_running = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = None


def parse_file_mentions(text: str) -> List[str]:
    """Extract @file mentions from text."""
    pattern = r'@([^\s@]+)'
    return re.findall(pattern, text)


def remove_file_mentions(text: str) -> str:
    """Remove @file mentions from text."""
    return re.sub(r'@([^\s@]+)', '', text).strip()


def format_diff_html(diff_text: str) -> str:
    """Format unified diff as HTML with syntax highlighting."""
    lines = diff_text.split('\n')
    html_lines = []
    
    for line in lines:
        if line.startswith('+++') or line.startswith('---'):
            html_lines.append(f'<div class="diff-line header">{line}</div>')
        elif line.startswith('@@'):
            html_lines.append(f'<div class="diff-line header">{line}</div>')
        elif line.startswith('+'):
            html_lines.append(f'<div class="diff-line added">{line}</div>')
        elif line.startswith('-'):
            html_lines.append(f'<div class="diff-line removed">{line}</div>')
        else:
            html_lines.append(f'<div class="diff-line context">{line}</div>')
    
    return '\n'.join(html_lines)


async def refresh_models(api_key: str, base_url: str) -> List[str]:
    """Refresh available models from NIM."""
    try:
        models = await discover_models(api_key, base_url)
        return models
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return []


def render_sidebar():
    """Render the left sidebar with settings and workspace explorer."""
    settings_manager = st.session_state.settings_manager
    settings = settings_manager.settings
    
    with st.sidebar:
        st.title("🤖 NIM Coding Assistant")
        st.caption("Powered by NVIDIA NIM")
        
        # --- NIM API Settings ---
        with st.expander("🔧 NIM API Settings", expanded=True):
            api_key = st.text_input(
                "NVIDIA API Key",
                value=settings.nim.api_key.get_secret_value() if settings.nim.api_key else "",
                type="password",
                help="Get your API key from https://build.nvidia.com"
            )
            
            base_url = st.text_input(
                "Base URL",
                value=settings.nim.base_url,
                help="NIM API endpoint"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", use_container_width=True):
                    settings_manager.update_nim_settings(api_key, base_url)
                    st.success("Settings saved!")
            
            with col2:
                if st.button("🔄 Refresh Models", use_container_width=True):
                    if api_key:
                        with st.spinner("Fetching models..."):
                            models = asyncio.run(refresh_models(api_key, base_url))
                            settings_manager.update_available_models(models)
                            st.success(f"Found {len(models)} models!")
                    else:
                        st.error("Please enter API key first")
        
        # --- Model Selection ---
        with st.expander("🧠 Model Assignments", expanded=True):
            available_models = settings.available_models
            if not available_models:
                available_models = [
                    "nvidia/nemotron-3-ultra-550b-a55b",
                    "deepseek-ai/deepseek-v4-flash-0731",
                    "nvidia/nemotron-3.5-lightning-30b-a3b"
                ]
            
            planner_model = st.selectbox(
                "Planner Model",
                available_models,
                index=available_models.index(settings.models.planner_model) 
                if settings.models.planner_model in available_models else 0,
                help="Model for creating execution plans"
            )
            
            coder_model = st.selectbox(
                "Coder Model",
                available_models,
                index=available_models.index(settings.models.coder_model) 
                if settings.models.coder_model in available_models else 1,
                help="Model for code generation and editing"
            )
            
            verifier_model = st.selectbox(
                "Verifier Model",
                available_models,
                index=available_models.index(settings.models.verifier_model) 
                if settings.models.verifier_model in available_models else 2,
                help="Model for test verification and self-healing"
            )
            
            if st.button("💾 Save Model Assignments", use_container_width=True):
                settings_manager.update_model_assignments(planner_model, coder_model, verifier_model)
                st.success("Model assignments saved!")
        
        # --- Workspace Settings ---
        with st.expander("📁 Workspace", expanded=True):
            workspace_dir = st.text_input(
                "Workspace Directory",
                value=settings.sandbox.workspace_dir,
                help="Root directory for file operations"
            )
            
            auto_approve = st.checkbox(
                "Auto-approve terminal commands",
                value=settings.sandbox.auto_approve_commands,
                help="Run commands without confirmation"
            )
            
            command_timeout = st.number_input(
                "Command Timeout (seconds)",
                value=settings.sandbox.command_timeout,
                min_value=10,
                max_value=600,
                step=10
            )
            
            if st.button("💾 Save Workspace Settings", use_container_width=True):
                settings_manager.update_sandbox_settings(workspace_dir, auto_approve, command_timeout)
                st.success("Workspace settings saved!")
                
                # Reinitialize sandbox
                st.session_state.sandbox = WorkspaceSandbox(workspace_dir, auto_approve, command_timeout)
                st.session_state.workspace_files = []
                st.rerun()
        
        # --- Mode Selection ---
        st.divider()
        mode = st.radio(
            "Mode",
            ["Ask (Chat)", "Agent (Codex)"],
            index=0 if st.session_state.current_mode == AgentMode.ASK else 1,
            horizontal=True
        )
        st.session_state.current_mode = AgentMode.ASK if mode == "Ask (Chat)" else AgentMode.AGENT
        
        if st.session_state.current_mode == AgentMode.AGENT:
            max_iterations = st.slider(
                "Max Self-Heal Iterations",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum fix attempts for failing tests"
            )
        
        # --- File Explorer ---
        st.divider()
        st.subheader("📂 File Explorer")
        
        if st.session_state.sandbox is None and workspace_dir:
            st.session_state.sandbox = WorkspaceSandbox(workspace_dir, auto_approve, command_timeout)
        
        if st.session_state.sandbox:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.workspace_files = []
                st.rerun()
            
            # Show file tree
            if not st.session_state.workspace_files:
                try:
                    tree_output = st.session_state.sandbox.list_files(".")
                    st.session_state.workspace_files = tree_output
                except Exception as e:
                    st.error(f"Error loading files: {e}")
            
            if st.session_state.workspace_files:
                st.code(st.session_state.workspace_files, language=None)
        
        # --- Session Controls ---
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.harness:
                st.session_state.harness.clear_history()
            st.rerun()


def render_chat_message(msg: Dict[str, Any]):
    """Render a single chat message."""
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    reasoning = msg.get("reasoning")
    tool_calls = msg.get("tool_calls")
    tool_results = msg.get("tool_results")
    diff = msg.get("diff")
    step_type = msg.get("step_type")
    
    # Status badge for agent steps
    status_html = ""
    if step_type:
        status_class = f"status-{step_type}"
        status_html = f'<span class="status-badge {status_class}">{step_type}</span> '
    
    # Message container
    css_class = f"chat-message {role}"
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    
    # Header with role and status
    role_icon = {"user": "👤", "assistant": "🤖", "tool": "🔧", "system": "⚙️"}.get(role, "🤖")
    st.markdown(f"**{status_html}{role_icon} {role.title()}**")
    
    # Main content
    if content:
        st.markdown(content)
    
    # Reasoning block
    if reasoning:
        with st.expander("🧠 Reasoning", expanded=False):
            st.markdown(f'<div class="reasoning-block">{reasoning}</div>', unsafe_allow_html=True)
    
    # Tool calls
    if tool_calls:
        for tc in tool_calls:
            with st.expander(f"🔧 Tool Call: {tc['name']}", expanded=True):
                st.markdown(f'<div class="tool-call-block">', unsafe_allow_html=True)
                st.markdown(f'<div class="tool-call-args">{json.dumps(tc["arguments"], indent=2)}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Tool results
    if tool_results:
        for tr in tool_results:
            with st.expander(f"📤 Result: {tr['name']}", expanded=False):
                st.markdown(f'<div class="tool-call-result">{tr["content"]}</div>', unsafe_allow_html=True)
    
    # Diff viewer
    if diff:
        st.markdown("### 📝 Changes")
        diff_html = format_diff_html(diff)
        st.markdown(f'''
        <div class="diff-viewer">
            <div class="diff-header">
                <span>Unified Diff</span>
                <div>
                    <button onclick="acceptDiff()">Accept</button>
                    <button onclick="revertDiff()">Revert</button>
                </div>
            </div>
            <div class="diff-content">{diff_html}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Accept Changes", key=f"accept_{msg.get('id', '')}"):
                st.success("Changes accepted!")
        with col2:
            if st.button("↩️ Revert Changes", key=f"revert_{msg.get('id', '')}"):
                st.warning("Changes reverted!")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_chat_panel():
    """Render the main chat panel."""
    st.subheader("💬 Chat")
    
    # Chat history container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(msg)
    
    # Chat input
    if not st.session_state.agent_running:
        user_input = st.chat_input(
            "Ask a question or describe a task... (use @filename to reference files)",
            key="chat_input"
        )
        
        if user_input:
            # Parse file mentions
            mentioned_files = parse_file_mentions(user_input)
            clean_input = remove_file_mentions(user_input)
            
            # Add user message
            user_msg = {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_msg)
            
            # Run agent
            st.session_state.agent_running = True
            st.rerun()
    
    # Show running indicator
    if st.session_state.agent_running:
        with st.spinner("Agent is working..."):
            st.info("🤖 Agent is processing your request...")


async def run_agent_task(
    harness: AgentHarness,
    user_prompt: str,
    context_files: List[str],
    mode: AgentMode,
    max_iterations: int = 5
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run the agent and yield progress updates."""
    if mode == AgentMode.ASK:
        async for chunk in harness.run_ask_mode(user_prompt, context_files):
            yield {"type": "stream", "content": chunk}
        yield {"type": "done"}
    else:
        async for update in harness.run_agent_mode(user_prompt, context_files, max_iterations=max_iterations):
            yield update


def process_agent_response():
    """Process the agent response asynchronously."""
    # This will be called when agent_running is True
    # We need to run the async generator and update session state
    pass


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat panel
    render_chat_panel()
    
    # Process agent if running
    if st.session_state.agent_running and st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg["role"] == "user":
            user_prompt = last_msg["content"]
            context_files = parse_file_mentions(user_prompt)
            
            # Create or reuse harness
            settings = st.session_state.settings_manager.settings
            sandbox = st.session_state.sandbox or WorkspaceSandbox(
                settings.sandbox.workspace_dir,
                settings.sandbox.auto_approve_commands,
                settings.sandbox.command_timeout
            )
            
            # Run agent in background
            # Note: Streamlit doesn't support true async, so we use a workaround
            # In production, you'd use a proper async framework or background tasks
            
            # For now, show a placeholder
            assistant_msg = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "Agent execution would run here. In a full implementation, this would stream the multi-agent loop output.",
                "step_type": "plan",
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_msg)
            st.session_state.agent_running = False
            st.rerun()


if __name__ == "__main__":
    main()

    def replace_in_file(self, path: str, search: str, replace: str) -> str:
        """Replace exact block in file. Returns unified diff."""
        full_path = self._validate_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        old_content = full_path.read_text(encoding="utf-8")
        if search not in old_content:
            raise ValueError("Search block not found in file")
        
        new_content = old_content.replace(search, replace, 1)
        full_path.write_text(new_content, encoding="utf-8")
        
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        return "".join(diff)

    def run_bash(self, command: str) -> Dict[str, Any]:
        """Run shell command in workspace directory."""
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.workspace_dir,
            capture_output=True,
            text=True,
            timeout=45,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

# =============================================================================
# TOOL CALL PARSING
# =============================================================================

TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>')
