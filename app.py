"""
NVIDIA NIM Agentic Coding Assistant - Split-Pane IDE Workbench.

Features:
- Left Column (Copilot & Agent Orchestrator):
  - Chat stream with @file autocomplete pills
  - Real-time sub-agent status cards
  - Interactive "Approve / Reject Diff" action buttons
- Right Column (Tabbed IDE & Profiler Inspector):
  - Tab 1: Live Code & Diff Viewer (side-by-side syntax-highlighted diff + inline editor)
  - Tab 2: Sandbox Terminal (live scrollable stdout/stderr + interactive command input)
  - Tab 3: NeMo Profiler & Telemetry (token consumption, latency waterfall, redundant calls, ATIF export)
"""

import streamlit as st
import asyncio
import json
import re
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

from core import (
    SettingsManager, get_settings_manager,
    discover_models,
    AgentHarness, AgentMode, build_sandbox,
    create_repo_mapper,
)
from sandbox import SandboxManager, GitGuard
from core.telemetry import ProfilerData

# Page configuration - WIDE LAYOUT FOR SPLIT PANE
st.set_page_config(
    page_title="NIM Agentic Coding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Split-Pane IDE Workbench
st.markdown("""
<style>
    /* Main layout - remove padding for full-width columns */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Column styling */
    .left-column {
        background-color: #1e1e1e;
        border-right: 1px solid #3d3d3d;
        height: calc(100vh - 120px);
        overflow-y: auto;
        padding: 1rem;
    }
    
    .right-column {
        background-color: #1e1e1e;
        height: calc(100vh - 120px);
        overflow: hidden;
        display: flex;
        flex-direction: column;
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
    
    /* Sub-agent status cards */
    .subagent-card {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .subagent-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subagent-status {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .status-idle { background-color: #3d3d3d; color: #cccccc; }
    .status-running { background-color: #007acc; color: white; }
    .status-success { background-color: #4ec9b0; color: #1e1e1e; }
    .status-error { background-color: #f48771; color: white; }
    
    /* File mention pills */
    .file-pill {
        display: inline-block;
        background-color: #007acc;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: background 0.1s;
    }
    
    .file-pill:hover {
        background-color: #005a9e;
    }
    
    /* Approve/Reject buttons */
    .diff-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .btn-approve {
        background-color: #4ec9b0 !important;
        color: #1e1e1e !important;
        border: none !important;
    }
    
    .btn-reject {
        background-color: #f48771 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Tabbed IDE */
    .ide-tabs {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .tab-content {
        flex: 1;
        overflow: auto;
        padding: 1rem;
    }
    
    /* Terminal */
    .terminal-output {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem;
        font-family: 'Monospace', monospace;
        font-size: 0.85rem;
        color: #e6edf3;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }
    
    .terminal-input {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.5rem;
        margin-top: 0.5rem;
        display: flex;
        gap: 0.5rem;
    }
    
    .terminal-prompt {
        color: #7ee787;
        font-family: 'Monospace', monospace;
        white-space: nowrap;
    }
    
    /* Profiler charts */
    .profiler-section {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .profiler-title {
        color: #9cdcfe;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #3d3d3d;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4ec9b0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #cccccc;
        margin-top: 0.25rem;
    }
    
    /* Redundant call warning */
    .redundant-warning {
        background-color: #4a2d2d;
        border: 1px solid #f48771;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #f48771;
    }
    
    /* Export button */
    .export-btn {
        background-color: #007acc !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: bold !important;
        border-radius: 6px !important;
    }
    
    /* Streamlit overrides */
    .stChatInputContainer {
        background-color: #1e1e1e;
        border-top: 1px solid #3d3d3d;
        padding: 1rem;
    }
    
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        color: #e6edf3;
    }
    
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
    }
    
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
    
    /* Code editor */
    .code-editor {
        background-color: #1e1e1e;
        border: 1px solid #3d3d3d;
        border-radius: 6px;
        overflow: hidden;
    }
    
    /* File tree */
    .file-tree {
        font-family: 'Monospace', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
    }
    
    .file-tree-item {
        padding: 0.2rem 0.5rem;
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
    .status-explore { background-color: #9cdcfe; color: #1e1e1e; }
</style>
""", unsafe_allow_html=True)


# ==================== Session state & resources ====================

def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        "messages": [],
        "mode": "ask",
        "sandbox": None,
        "git_guard": None,
        "resource_key": None,
        "agent_running": False,
        "last_run_checkpoint": None,
        "available_models": [],
        "notice": None,
        "current_diffs": [],
        "pending_diff_index": 0,
        "terminal_history": [],
        "terminal_input": "",
        "selected_file": None,
        "profiler_data": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "settings_manager" not in st.session_state:
        st.session_state.settings_manager = get_settings_manager()


def _cleanup_sandbox():
    sb = st.session_state.get("sandbox")
    if sb is not None:
        try:
            sb.cleanup()
        except Exception:
            pass


def get_resources():
    """(Re)create SandboxManager and GitGuard when workspace/sandbox settings change."""
    s = st.session_state.settings_manager.settings.sandbox
    key = (s.workspace_dir, str(s.sandbox_type), s.docker_image, s.command_timeout,
           s.auto_approve_commands, s.docker_cpu_limit, s.docker_memory_limit, s.auto_git_checkpoint)
    if st.session_state.resource_key != key:
        _cleanup_sandbox()
        Path(s.workspace_dir).mkdir(parents=True, exist_ok=True)
        st.session_state.sandbox = build_sandbox(st.session_state.settings_manager)
        st.session_state.git_guard = None
        if s.auto_git_checkpoint:
            try:
                st.session_state.git_guard = GitGuard(s.workspace_dir)
            except Exception as e:
                st.session_state.notice = ("warning", f"Git checkpoints unavailable: {e}")
        st.session_state.resource_key = key
    return st.session_state.sandbox, st.session_state.git_guard


def make_harness(mode: AgentMode) -> AgentHarness:
    sandbox, guard = get_resources()
    return AgentHarness(
        sandbox=sandbox,
        settings_manager=st.session_state.settings_manager,
        mode=mode,
        git_guard=guard,
    )


# ==================== Rendering helpers ====================

def parse_file_mentions(text: str) -> List[str]:
    """Extract @file mentions from text."""
    return re.findall(r"@([\w\-./\\]+)", text)


def format_diff_html(diff_text: str) -> str:
    """Render a unified diff as HTML."""
    import html
    lines = []
    for line in diff_text.splitlines():
        cls = "context"
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            cls = "header"
        elif line.startswith("+"):
            cls = "added"
        elif line.startswith("-"):
            cls = "removed"
        lines.append(f'<div class="diff-line {cls}">{html.escape(line)}</div>')
    return f'<div class="diff-viewer"><div class="diff-content">{"".join(lines)}</div></div>'


def render_chat_message(msg: Dict[str, Any]):
    role = msg.get("role", "assistant")
    with st.chat_message("user" if role == "user" else "assistant"):
        if msg.get("step"):
            step = msg['step']
            badge_class = f"status-{step}" if step in ["plan", "edit", "verify", "fix", "explore"] else "status-plan"
            st.markdown(f'<span class="status-badge {badge_class}">{step.upper()}</span>', unsafe_allow_html=True)
        if msg.get("content"):
            st.markdown(msg["content"])
        for tc in msg.get("tool_events", []):
            with st.expander(f"🔧 {tc['name']}", expanded=False):
                if tc.get("args"):
                    st.code(json.dumps(tc["args"], indent=2)[:3000], language="json")
                if tc.get("result") is not None:
                    st.code(str(tc["result"])[:4000])


def render_subagent_status_cards(harness: AgentHarness):
    """Render real-time sub-agent status cards."""
    status = harness.get_sub_agent_status()
    if not status:
        return
    
    st.markdown("### 🤖 Sub-Agent Status")
    cols = st.columns(len(status))
    for i, (agent, state) in enumerate(status.items()):
        with cols[i]:
            status_class = f"status-{state}" if state in ["idle", "running", "success", "error"] else "status-idle"
            icon = {"explorer": "🔍", "coder": "💻", "test_writer": "🧪"}.get(agent, "🤖")
            st.markdown(f"""
            <div class="subagent-card">
                <div class="subagent-header">
                    <span>{icon}</span>
                    <strong>{agent.replace('_', ' ').title()}</strong>
                    <span class="subagent-status {status_class}">{state}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_diff_viewer():
    """Render the live code & diff viewer tab."""
    st.markdown("### 📝 Live Code & Diff Viewer")
    
    if not st.session_state.current_diffs:
        st.info("No diffs available. Run an agent task to see changes.")
        return
    
    # File selector
    diff_files = [d.get("file", "unknown") for d in st.session_state.current_diffs]
    selected_idx = st.selectbox(
        "Select file",
        range(len(diff_files)),
        format_func=lambda i: diff_files[i],
        index=st.session_state.pending_diff_index,
        key="diff_file_selector"
    )
    st.session_state.pending_diff_index = selected_idx
    
    diff = st.session_state.current_diffs[selected_idx]
    
    # Diff view
    st.markdown(f"**File:** `{diff.get('file', 'unknown')}` | **Status:** {diff.get('status', 'unknown')}")
    
    diff_html = format_diff_html(diff.get("diff", ""))
    st.markdown(diff_html, unsafe_allow_html=True)
    
    # Approve/Reject buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ Approve", key=f"approve_{selected_idx}", type="primary", use_container_width=True):
            st.success(f"Approved changes to {diff.get('file')}")
    with col2:
        if st.button("❌ Reject", key=f"reject_{selected_idx}", type="secondary", use_container_width=True):
            st.warning(f"Rejected changes to {diff.get('file')}")
    with col3:
        if st.button("📋 Copy Diff", key=f"copy_{selected_idx}", use_container_width=True):
            st.code(diff.get("diff", ""), language="diff")


def render_terminal_tab(sandbox: SandboxManager):
    """Render the sandbox terminal tab."""
    st.markdown("### 🖥️ Sandbox Terminal")
    
    # Terminal output
    output_container = st.container()
    with output_container:
        if st.session_state.terminal_history:
            terminal_text = "\n".join(st.session_state.terminal_history[-100:])  # Last 100 lines
            st.markdown(f'<div class="terminal-output">{terminal_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="terminal-output">Terminal ready. Type a command below.</div>', unsafe_allow_html=True)
    
    # Command input
    col1, col2 = st.columns([10, 1])
    with col1:
        cmd = st.text_input(
            "Command",
            value=st.session_state.terminal_input,
            placeholder="Enter command (e.g., ls -la, pytest -v, python script.py)",
            key="terminal_cmd_input",
            label_visibility="collapsed"
        )
    with col2:
        if st.button("▶ Run", key="terminal_run", use_container_width=True):
            if cmd.strip():
                st.session_state.terminal_history.append(f"$ {cmd}")
                try:
                    result = sandbox.run_terminal(cmd)
                    output = f"Exit code: {result.exit_code}\n{result.stdout}"
                    if result.stderr:
                        output += f"\n{result.stderr}"
                    if result.timed_out:
                        output += f"\n[Timed out after {result.duration_ms}ms]"
                    st.session_state.terminal_history.append(output)
                except Exception as e:
                    st.session_state.terminal_history.append(f"Error: {e}")
                st.session_state.terminal_input = ""
                st.rerun()
    
    # Quick command buttons
    st.caption("Quick commands:")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        if st.button("ls -la", key="q_ls", use_container_width=True):
            st.session_state.terminal_input = "ls -la"
            st.rerun()
    with qcol2:
        if st.button("pytest -v", key="q_pytest", use_container_width=True):
            st.session_state.terminal_input = "pytest -v"
            st.rerun()
    with qcol3:
        if st.button("git status", key="q_git", use_container_width=True):
            st.session_state.terminal_input = "git status"
            st.rerun()
    with qcol4:
        if st.button("Clear", key="q_clear", use_container_width=True):
            st.session_state.terminal_history = []
            st.rerun()


def render_profiler_tab(harness: AgentHarness):
    """Render the NeMo Profiler & Telemetry tab."""
    st.markdown("### 📊 NeMo Profiler & Telemetry")
    
    profiler_data = harness.get_telemetry_data()
    if not profiler_data:
        st.info("No telemetry data available. Run an agent task to collect metrics.")
        if st.button("🔄 Refresh", key="profiler_refresh"):
            st.rerun()
        return
    
    plotly_data = profiler_data.to_plotly_data()
    
    # Summary metrics
    st.markdown("#### 📈 Run Summary")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{plotly_data['llm_call_count']}</div>
            <div class="metric-label">LLM Calls</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{plotly_data['tool_call_count']}</div>
            <div class="metric-label">Tool Calls</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{plotly_data['total_tokens']:,}</div>
            <div class="metric-label">Total Tokens</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{plotly_data['duration_ms']:.0f}ms</div>
            <div class="metric-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Token consumption by role
    st.markdown("#### 🎯 Token Consumption by Agent Role")
    tokens_by_role = plotly_data.get("tokens_by_role", {})
    if tokens_by_role:
        role_cols = st.columns(len(tokens_by_role))
        for i, (role, tokens) in enumerate(tokens_by_role.items()):
            with role_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{tokens['total']:,}</div>
                    <div class="metric-label">{role.title()}</div>
                    <div style="font-size: 0.7rem; color: #888; margin-top: 0.5rem;">
                        Prompt: {tokens['prompt']:,} | Completion: {tokens['completion']:,} | Reasoning: {tokens['reasoning']:,}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Latency waterfall
    st.markdown("#### ⏱️ Latency Waterfall")
    waterfall = plotly_data.get("latency_waterfall", [])
    if waterfall:
        for event in waterfall:
            if event["type"] == "llm":
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.5rem; background: #2d2d2d; border-radius: 4px; margin: 0.25rem 0;">
                    <span style="color: #9cdcfe; font-weight: bold; width: 100px;">{event['role']}</span>
                    <span style="color: #4ec9b0; width: 150px;">{event['model'][:30]}</span>
                    <div style="flex: 1; background: #1e1e1e; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="width: {min(event['duration_ms'] / plotly_data['duration_ms'] * 100, 100)}%; height: 100%; background: linear-gradient(90deg, #007acc, #4ec9b0);"></div>
                    </div>
                    <span style="color: #dcdcaa; width: 100px; text-align: right;">{event['duration_ms']:.0f}ms</span>
                    <span style="color: #888; width: 80px; text-align: right;">{event['tokens']:,} tok</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.5rem; background: #2d2d2d; border-radius: 4px; margin: 0.25rem 0;">
                    <span style="color: #ce9178; font-weight: bold; width: 100px;">TOOL</span>
                    <span style="color: #dcdcaa; width: 150px;">{event['name']}</span>
                    <div style="flex: 1; background: #1e1e1e; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="width: {min(event['duration_ms'] / plotly_data['duration_ms'] * 100, 100)}%; height: 100%; background: linear-gradient(90deg, #ce9178, #dcdcaa);"></div>
                    </div>
                    <span style="color: {'#4ec9b0' if event['success'] else '#f48771'}; width: 100px; text-align: right;">{event['duration_ms']:.0f}ms</span>
                    <span style="color: #888; width: 80px;"></span>
                </div>
                """, unsafe_allow_html=True)
    
    # Redundant call warnings
    st.markdown("#### ⚠️ Redundant Call Warnings")
    redundant = plotly_data.get("redundant_calls", [])
    if redundant:
        for warn in redundant:
            st.markdown(f"""
            <div class="redundant-warning">
                <strong>⚠️ Redundant call detected:</strong> <code>{warn['tool_name']}</code> called {warn['call_count']} times with identical arguments<br>
                First: {warn['first_timestamp']} | Last: {warn['last_timestamp']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No redundant calls detected ✓")
    
    # Export trajectory
    st.markdown("#### 📤 Export Trajectory")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Export the completed run as ATIF JSONL for model fine-tuning or evaluation.")
    with col2:
        if st.button("📤 Export Verified Trajectory (JSONL)", key="export_trajectory", type="primary", use_container_width=True):
            if profiler_data and hasattr(harness, '_telemetry_context') and harness._telemetry_context:
                try:
                    path = asyncio.run(harness._telemetry_context.export_trajectory_async())
                    st.success(f"Exported to: {path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
            else:
                st.warning("No trajectory to export")


def render_repo_map_modal():
    """Render repo map in a modal/expander."""
    sm = st.session_state.settings_manager
    query = st.text_input("Rank by keywords (optional)", key="repo_map_query")
    budget = st.slider("Token budget", 500, 8000, sm.settings.repo_map.token_budget, 250, key="repo_map_budget")
    sandbox, _ = get_resources()
    try:
        harness = make_harness(AgentMode.ASK)
        text = harness.get_repo_map(query=query, force_refresh=True, max_tokens=budget)
        stats = harness.get_repo_map_stats()
    except Exception as e:
        st.error(f"Failed to build repo map: {e}")
        return
    if stats:
        st.caption(" · ".join(f"{k}: {v}" for k, v in stats.items() if isinstance(v, (int, float, str))))
    st.code(text or "(empty)", language="xml")


if hasattr(st, "dialog"):
    @st.dialog("🗺️ Live Repo Map", width="large")
    def show_repo_map():
        render_repo_map_modal()
else:
    def show_repo_map():
        with st.expander("🗺️ Live Repo Map", expanded=True):
            render_repo_map_modal()


# ==================== Sidebar ====================

def render_docker_status(sandbox):
    status = sandbox.get_container_status()
    st_name = status.get("status", "N/A")
    if status.get("type") != "docker":
        st.markdown("🖥️ **Sandbox:** local subprocess")
        return
    icon = {"running": "🟢", "stopped": "🔴", "unhealthy": "🟡"}.get(st_name, "⚪")
    cid = (status.get("container_id") or "")[:12]
    st.markdown(f"{icon} **Docker:** {st_name}" + (f" (`{cid}`)" if cid else ""))
    c1, c2 = st.columns(2)
    if c1.button("Start", key="docker_start", use_container_width=True):
        with st.spinner("Starting container..."):
            ok = sandbox.start_container()
        st.session_state.notice = ("success", "Container started") if ok else ("error", "Failed to start container (is Docker running?)")
        st.rerun()
    if c2.button("Stop", key="docker_stop", use_container_width=True):
        sandbox.stop_container()
        st.rerun()


def render_sidebar():
    sm = st.session_state.settings_manager
    settings = sm.settings
    with st.sidebar:
        st.title("⚙️ Settings")

        st.session_state.mode = st.radio(
            "Mode", ["ask", "agent"], index=0 if st.session_state.mode == "ask" else 1,
            format_func=lambda m: "💬 Ask" if m == "ask" else "🤖 Agent",
            horizontal=True,
        )

        with st.expander("🔑 NIM API", expanded=not settings.nim.api_key.get_secret_value()):
            api_key = st.text_input("API Key", value=settings.nim.api_key.get_secret_value(), type="password")
            base_url = st.text_input("Base URL", value=settings.nim.base_url)
            if st.button("Save API settings"):
                sm.update_nim_settings(api_key, base_url)
                st.success("Saved")
            if st.button("🔍 Discover models"):
                try:
                    models = asyncio.run(discover_models(api_key, base_url))
                    sm.update_available_models(models)
                    st.success(f"Found {len(models)} models")
                except Exception as e:
                    st.error(f"Discovery failed: {e}")

        with st.expander("🧠 Models"):
            options = settings.available_models or [
                settings.models.planner_model, settings.models.coder_model, settings.models.verifier_model]
            options = list(dict.fromkeys(options))

            def pick(label, current):
                opts = options if current in options else [current] + options
                return st.selectbox(label, opts, index=opts.index(current))
            planner = pick("Planner", settings.models.planner_model)
            coder = pick("Coder", settings.models.coder_model)
            verifier = pick("Verifier", settings.models.verifier_model)
            if st.button("Save models"):
                sm.update_model_assignments(planner, coder, verifier)
                st.success("Saved")

        with st.expander("📁 Workspace & Sandbox", expanded=True):
            ws = st.text_input("Workspace directory", value=settings.sandbox.workspace_dir)
            sb_type = st.selectbox(
                "Sandbox type", ["local", "docker"],
                index=0 if str(getattr(settings.sandbox.sandbox_type, "value", settings.sandbox.sandbox_type)) == "local" else 1)
            image = st.text_input("Docker image", value=settings.sandbox.docker_image)
            auto_approve = st.checkbox("Auto-approve commands", value=settings.sandbox.auto_approve_commands)
            auto_git = st.checkbox("Auto Git checkpoint before agent runs", value=settings.sandbox.auto_git_checkpoint)
            timeout = st.number_input("Command timeout (s)", 5, 3600, int(settings.sandbox.command_timeout))
            if st.button("Apply sandbox settings"):
                sm.update_sandbox_settings(
                    workspace_dir=ws, auto_approve=auto_approve, command_timeout=int(timeout),
                    sandbox_type=sb_type, docker_image=image, auto_git_checkpoint=auto_git)
                st.rerun()

        sandbox, guard = get_resources()
        st.subheader("🐳 Sandbox Status")
        render_docker_status(sandbox)

        st.subheader("🛡️ Git Safety")
        if guard is None:
            st.caption("Auto Git checkpoints are disabled.")
        else:
            cp = guard.get_active_checkpoint()
            st.caption(f"Active checkpoint: `{cp.checkpoint_id}`" if cp else "No active checkpoint")
            if st.button("⏮ Rollback Last Agent Run", type="primary", disabled=cp is None,
                         use_container_width=True, key="rollback_btn"):
                result = guard.rollback_last()
                if result.get("success"):
                    st.session_state.notice = ("success", "Rolled back the last agent run.")
                else:
                    st.session_state.notice = ("error", result.get("error", "Rollback failed"))
                st.rerun()
            msg = st.text_input("Commit message", value="Accept agent changes", key="commit_msg")
            if st.button("✅ Accept & Commit Changes", use_container_width=True, key="commit_btn"):
                result = guard.commit_accepted_changes(msg)
                if result.get("success"):
                    st.session_state.notice = ("success", f"Committed: {result.get('message')}")
                else:
                    st.session_state.notice = ("error", result.get("error", "Commit failed"))
                st.rerun()

        st.subheader("🔌 MCP Servers")
        harness = make_harness(AgentMode.ASK)
        mcp_status = harness.get_mcp_status()
        if mcp_status:
            for name, status in mcp_status.items():
                icon = "🟢" if status.get("healthy") else "🔴"
                st.markdown(f"{icon} **{name}** ({status.get('transport', '?')}) - {status.get('tool_count', 0)} tools")
                if not status.get("healthy") and status.get("enabled"):
                    if st.button(f"🔄 Restart {name}", key=f"restart_mcp_{name}", use_container_width=True):
                        asyncio.run(harness._mcp_manager.restart_server(name))
                        st.rerun()
        else:
            st.caption("No MCP servers configured. Enable in settings.json")

        st.subheader("🪝 Hooks")
        hook_status = harness.get_hook_status()
        if hook_status.get("enabled"):
            st.success("Hooks enabled")
            st.caption(f"Pre-hooks: {', '.join(hook_status.get('pre_hooks', []))}")
            st.caption(f"Post-hooks: {', '.join(hook_status.get('post_hooks', []))}")
        else:
            st.caption("Hooks disabled")

        st.subheader("🗺️ Context")
        if st.button("🗺️ View Live Repo Map", use_container_width=True, key="repo_map_btn"):
            show_repo_map()

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_diffs = []
            st.session_state.terminal_history = []
            st.rerun()


# ==================== Agent execution ====================

async def run_agent_turn(prompt: str, mode: str, placeholder_container):
    """Run a single user turn, streaming into the UI. Returns the assistant message dict."""
    files = parse_file_mentions(prompt)
    out: Dict[str, Any] = {"role": "assistant", "content": "", "tool_events": []}
    live = placeholder_container.empty()
    pending: Dict[str, Dict[str, Any]] = {}

    def refresh(text: str):
        live.markdown(text + " ▌")

    async with make_harness(AgentMode.ASK if mode == "ask" else AgentMode.AGENT) as harness:
        # Set prompt for telemetry
        if harness._telemetry_context:
            harness._telemetry_context.set_prompt(prompt)
        
        if mode == "ask":
            async for chunk in harness.run_ask_mode(prompt, files):
                out["content"] += chunk
                refresh(out["content"])
        else:
            # Use sub-agent harness for agent mode
            async for ev in harness._sub_agent_harness.run_agent_mode_with_subagents(prompt, files):
                t = ev.get("type")
                if t == "step_start":
                    step = ev.get('step', '')
                    phase = ev.get('phase', '')
                    out["content"] += f"\n\n**▶ {step.upper()}** ({phase})\n\n"
                elif t == "stream":
                    out["content"] += ev.get("content", "")
                elif t == "tool_call":
                    entry = {"name": ev.get("name"), "args": ev.get("arguments"), "result": None}
                    out["tool_events"].append(entry)
                    out["content"] += f"\n\n🔧 `{ev.get('name')}`\n\n"
                elif t == "tool_result":
                    if out["tool_events"]:
                        out["tool_events"][-1]["result"] = ev.get("content")
                elif t == "checkpoint":
                    st.session_state.last_run_checkpoint = ev.get("checkpoint_id")
                    out["content"] += f"\n\n🛡️ Checkpoint `{ev.get('checkpoint_id')}` created.\n\n"
                elif t == "warning":
                    out["content"] += f"\n\n⚠️ {ev.get('message') or ev.get('content')}\n\n"
                elif t == "success":
                    out["content"] += f"\n\n✅ {ev['message']}\n"
                elif t == "failure":
                    out["content"] += f"\n\n❌ {ev['message']}\n"
                elif t == "step_complete":
                    step = ev.get("step", "")
                    phase = ev.get("phase", "")
                    success = ev.get("success", True)
                    summary = ev.get("summary", "")
                    files_modified = ev.get("files_modified", [])
                    diffs = ev.get("diffs", [])
                    
                    if diffs:
                        st.session_state.current_diffs.extend(diffs)
                    
                    status_icon = "✅" if success else "❌"
                    out["content"] += f"\n\n{status_icon} **{step.upper()}** ({phase}) complete\n"
                    if summary:
                        out["content"] += f"{summary[:500]}\n"
                    if files_modified:
                        out["content"] += f"Files: {', '.join(files_modified)}\n"
                refresh(out["content"])
    
    live.markdown(out["content"])
    return out


# ==================== Main Layout ====================

def main():
    init_session_state()
    
    # Check API key
    if not st.session_state.settings_manager.settings.nim.api_key.get_secret_value():
        st.error("⚠️ Set your NIM API key in the sidebar first.")
        render_sidebar()
        return
    
    # Get resources
    sandbox, guard = get_resources()
    harness = make_harness(AgentMode.ASK if st.session_state.mode == "ask" else AgentMode.AGENT)
    
    # SPLIT PANE LAYOUT
    col_left, col_right = st.columns([0.55, 0.45], gap="large")
    
    # ========== LEFT COLUMN: COPILOT & AGENT ORCHESTRATOR ==========
    with col_left:
        st.markdown('<div class="left-column">', unsafe_allow_html=True)
        
        st.title("🤖 NIM Agentic Coding Assistant")
        
        notice = st.session_state.notice
        if notice:
            getattr(st, notice[0], st.info)(notice[1])
            st.session_state.notice = None
        
        # Sub-agent status cards
        if st.session_state.mode == "agent":
            render_subagent_status_cards(harness)
            st.divider()
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                render_chat_message(msg)
        
        # Chat input
        prompt = st.chat_input("Ask about your code, or describe a change. Use @file to attach files.")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                container = st.container()
                try:
                    result = asyncio.run(run_agent_turn(prompt, st.session_state.mode, container))
                except Exception as e:
                    result = {"role": "assistant", "content": f"❌ Error: {e}", "tool_events": []}
                    container.error(result["content"])
            st.session_state.messages.append(result)
            if st.session_state.mode == "agent":
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== RIGHT COLUMN: TABBED IDE & PROFILER ==========
    with col_right:
        st.markdown('<div class="right-column">', unsafe_allow_html=True)
        
        # Tab navigation
        tab1, tab2, tab3 = st.tabs(["📝 Code & Diff", "🖥️ Terminal", "📊 Profiler"])
        
        with tab1:
            render_diff_viewer()
        
        with tab2:
            render_terminal_tab(sandbox)
        
        with tab3:
            render_profiler_tab(harness)
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()