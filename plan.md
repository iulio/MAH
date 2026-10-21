# NVIDIA NIM Agentic Coding Harness V1.0 Production Release Plan

## Problem Statement
Deliver production-ready V1.0 release with:
1. Dynamic Skills Engine (SKILL.md standard)
2. Full MCP Server Hub & Tool Router
3. Final UX Polish (multi-session, project rules, slash commands)
4. Headless FastAPI Daemon & CLI
5. Automated Benchmark & Regression Evaluator
6. Zero-friction Docker packaging

All code must be complete, modular, runnable, zero placeholders/TODOs.

## Current State Analysis
- Existing Streamlit workbench in `app.py` (42KB, split-pane IDE)
- Core modules: agent_orchestrator.py, config.py, context_manager.py, hooks.py, mcp_bridge.py, repo_map.py, telemetry.py
- Existing MCP bridge with stdio/sse support but no lifecycle hub
- No skills_manager, session_store, server.py, cli.py, evaluator
- Requirements: streamlit, openai, pydantic, pathspec, tiktoken, httpx, python-dotenv, watchdog, rich, pygments
- Settings.json has nim, models, sandbox, mcp servers, telemetry, hooks

## Implementation Todos

### Phase 1: Dynamic Skills Engine
- [ ] core/skills_manager.py
  - Scan ./skills/<skill-name>/SKILL.md + scripts/
  - Parse YAML frontmatter (name, description, triggers, recommended_model)
  - Tier 1: inject compact skill list into system prompts
  - Tier 2: load_skill(skill_name) tool + /skill-name slash command
  - create_skill_from_trajectory(name, description, instructions)
  - Auto-seed tdd-python, security-audit, pr-packager if ./skills empty

### Phase 2: Full MCP Server Hub
- [ ] core/mcp_hub.py
  - Config mcp_servers.json with name, transport, command/args/env/url, enabled
  - Preset templates: Filesystem, GitHub, PostgreSQL/SQLite, Fetch/Web Search, Docker
  - Lifecycle: connect_server(), list_server_tools(), call_mcp_tool(), health ping
  - Namespace tools as mcp__<server>__<tool>
  - Permission policy: always_allow/ask_first/blocked with UI confirmation

### Phase 3: Session Store & UX Polish
- [ ] core/session_store.py
  - Persistent multi-session history in ./sessions/<session_id>.json via SQLite/JSON
  - Save/load chat threads, diffs, token telemetry
  - Project rules detection: HARNESS.md, AGENTS.md, .cursorrules injection
  - Slash commands: /skill, /test, /undo, /compact, @path
  - Update app.py tabs 4-5: Skills Library, MCP Server Hub

### Phase 4: Headless API & CLI
- [ ] server.py (FastAPI)
  - POST /v1/run with SSE streaming
  - POST /v1/evaluate
  - GET /health (NIM, models, Docker, MCP)
- [ ] cli.py
  - run, serve, eval commands

### Phase 5: Benchmark Evaluator
- [ ] core/evaluator.py
  - Generate evals/coding_tasks.json starter suite (3 tasks)
  - Isolated workspace runner, verification_command execution
  - Metrics: pass_at_1, healing_iterations_used, total_tokens, p50/p95 latency, redundant_tool_calls
  - Leaderboard UI tab in app.py

### Phase 6: Packaging
- [ ] pyproject.toml / requirements.txt update
- [ ] Dockerfile & docker-compose.yml (harness-ui:8501, harness-api:8080)
- [ ] Makefile / start.sh with setup/ui/api/test targets

## Notes
- Windows paths with backslashes
- Zero placeholders/TODOs required
- Preserve existing behavior
- Use existing core modules where possible
- Skills frontmatter parsing needs PyYAML
- MCP hub extends existing mcp_bridge.py
- Session store should be SQLite for performance
- FastAPI SSE streaming for /v1/run
- Docker volumes: ./workspace, ./skills, ./sessions, ./trajectories, settings.json
