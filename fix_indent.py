with open('core/agent_orchestrator.py', 'r') as f:
    lines = f.readlines()

# Fix lines 236-240 (0-indexed: 235-239)
lines[235] = '    @staticmethod\n'
lines[236] = '    def _read_only_tools() -> List[Dict[str, Any]]:\n'
lines[237] = '        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in READ_ONLY_TOOLS]\n'
lines[238] = '\n'
lines[239] = '    def _build_system_prompt(self, role: str) -> str:\n'

with open('core/agent_orchestrator.py', 'w') as f:
    f.writelines(lines)

print('Fixed!')