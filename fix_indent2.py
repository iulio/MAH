with open('core/agent_orchestrator.py', 'r') as f:
    lines = f.readlines()

# Fix lines 540-545 (0-indexed: 539-544)
lines[539] = '        # Track LLM call for telemetry\n'
lines[540] = '        llm_start_time = time.time()\n'
lines[541] = '        prompt_tokens_estimate = 0  # Would need tiktoken for accurate count\n'
lines[542] = '\n'
lines[543] = '        for _ in range(MAX_TOOL_ROUNDS):\n'

with open('core/agent_orchestrator.py', 'w') as f:
    f.writelines(lines)

print('Fixed!')