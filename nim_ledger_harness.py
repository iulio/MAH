import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

class ContextBlockLedger:
    """Manages the immutable SQLite blockchain of evicted context windows."""
    def __init__(self, db_path: str = "nim_context_ledger.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS context_chain (
                    block_id TEXT PRIMARY KEY,
                    prev_hash TEXT NOT NULL,
                    block_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    semantic_header TEXT NOT NULL,
                    raw_messages TEXT NOT NULL
                )
            """)

    def get_tip(self) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM context_chain ORDER BY rowid DESC LIMIT 1")
        return cur.fetchone()

    def mint_block(self, semantic_header: str, evicted_messages: List[Dict[str, Any]]) -> Dict[str, str]:
        tip = self.get_tip()
        prev_hash = tip["block_hash"] if tip else "0" * 64
        
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM context_chain")
        next_idx = cur.fetchone()["cnt"] + 1
        block_id = f"block_{next_idx:04d}"
        
        timestamp = datetime.now(timezone.utc).isoformat()
        raw_json = json.dumps(evicted_messages, ensure_ascii=False)
        
        # Cryptographic link + semantic state commitment
        payload = f"{block_id}|{prev_hash}|{timestamp}|{semantic_header}|{raw_json}"
        block_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self.conn:
            self.conn.execute(
                """INSERT INTO context_chain 
                   (block_id, prev_hash, block_hash, timestamp, semantic_header, raw_messages)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (block_id, prev_hash, block_hash, timestamp, semantic_header, raw_json)
            )
        return {
            "block_id": block_id,
            "prev_hash": prev_hash[:12],
            "block_hash": block_hash[:12],
            "semantic_header": semantic_header
        }

    def get_block_payload(self, block_id: str) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM context_chain WHERE block_id = ?", (block_id,))
        row = cur.fetchone()
        if not row:
            return json.dumps({"error": f"Block {block_id} not found in ledger."})
        return json.dumps({
            "block_id": row["block_id"],
            "prev_hash": row["prev_hash"],
            "timestamp": row["timestamp"],
            "semantic_header": row["semantic_header"],
            "raw_messages": json.loads(row["raw_messages"])
        }, indent=2)

    def get_chain_index(self, limit: int = 5) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT block_id, prev_hash, block_hash, semantic_header FROM context_chain ORDER BY rowid DESC LIMIT ?",
            (limit,)
        )
        return [dict(r) for r in reversed(cur.fetchall())]


class NIMLedgerHarness:
    """Wraps NVIDIA NIM chat completions with automatic blockchain context paging."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        primary_model: str = "meta/llama-3.3-70b-instruct",
        compressor_model: str = "meta/llama-3.1-8b-instruct",
        max_active_tokens: int = 6000,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("NIM_API_KEY"),
            base_url=base_url
        )
        self.primary_model = primary_model
        self.compressor_model = compressor_model
        self.max_active_tokens = max_active_tokens
        self.ledger = ContextBlockLedger()
        
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are an assistant connected to a Blockchain Context Ledger. "
                "Older conversation turns are sealed into immutable blocks. "
                "Read the [CONTEXT LEDGER HEAD] for the rolling state and block directory. "
                "If you need exact code, logs, or verbatim details from an older block, "
                "call `inspect_context_block` with its `block_id`."
            )
        }
        self.active_messages: List[Dict[str, Any]] = []
        self.last_prompt_tokens: int = 0

        self.ledger_tools = [
            {
                "type": "function",
                "function": {
                    "name": "inspect_context_block",
                    "description": "Dereference a sealed context block ID from the ledger to retrieve verbatim historical messages and code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "block_id": {
                                "type": "string",
                                "description": "The block identifier, e.g. 'block_0001'"
                            }
                        },
                        "required": ["block_id"]
                    }
                }
            }
        ]

    async def _compress_to_semantic_header(self, evicted: List[Dict[str, Any]], prev_header: str) -> str:
        """Uses a fast NIM model to fold evicted messages + previous state into a new block header."""
        prompt = (
            f"Previous Block State:\n{prev_header}\n\n"
            f"Evicted Conversation Segment:\n{json.dumps(evicted)}\n\n"
            "Produce a dense Semantic Block Header in JSON with keys: "
            "`rolling_world_state` (cumulative project state, configs, constraints), "
            "`block_delta` (what specifically happened in this evicted segment), and "
            "`artifacts_index` (list of functions, files, or decisions inside this block so the LLM knows when to dereference it)."
        )
        resp = await self.client.chat.completions.create(
            model=self.compressor_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )
        return resp.choices[0].message.content.strip()

    async def _maybe_mint_block(self):
        """Checks if the active window is approaching saturation and dumps oldest 60% into a linked block."""
        if self.last_prompt_tokens < self.max_active_tokens and len(self.active_messages) < 12:
            return

        # Slice oldest 60% of conversational turns (keep recent tail intact)
        split_idx = max(2, int(len(self.active_messages) * 0.6))
        evicted = self.active_messages[:split_idx]
        self.active_messages = self.active_messages[split_idx:]

        tip = self.ledger.get_tip()
        prev_header = tip["semantic_header"] if tip else "Genesis Block (No prior state)"

        semantic_header = await self._compress_to_semantic_header(evicted, prev_header)
        minted = self.ledger.mint_block(semantic_header, evicted)
        print(f"[LEDGER] Minted {minted['block_id']} | Hash: {minted['block_hash']} | Prev: {minted['prev_hash']}")

    def _build_window_payload(self) -> List[Dict[str, Any]]:
        """Constructs the active window: System Prompt + Ledger Head + Active Tail."""
        chain_index = self.ledger.get_chain_index(limit=5)
        if not chain_index:
            return [self.system_prompt] + self.active_messages

        ledger_banner = {
            "role": "system",
            "content": (
                "=== [CONTEXT LEDGER HEAD] ===\n"
                + "\n".join(
                    f"[{b['block_id']} | prev:{b['prev_hash'][:8]} -> hash:{b['block_hash'][:8]}] {b['semantic_header']}"
                    for b in chain_index
                )
                + "\n=== [END LEDGER HEAD] ==="
            )
        }
        return [self.system_prompt, ledger_banner] + self.active_messages

    async def chat(self, user_input: str) -> str:
        await self._maybe_mint_block()
        self.active_messages.append({"role": "user", "content": user_input})

        while True:
            payload_messages = self._build_window_payload()
            response = await self.client.chat.completions.create(
                model=self.primary_model,
                messages=payload_messages,
                tools=self.ledger_tools,
                tool_choice="auto"
            )

            if response.usage:
                self.last_prompt_tokens = response.usage.prompt_tokens

            msg = response.choices[0].message

            # If the NIM model decides to walk the blockchain backwards:
            if msg.tool_calls:
                self.active_messages.append(msg.model_dump(exclude_none=True))
                for tc in msg.tool_calls:
                    if tc.function.name == "inspect_context_block":
                        args = json.loads(tc.function.arguments)
                        block_data = self.ledger.get_block_payload(args["block_id"])
                        print(f"[LEDGER] Dereferencing pointer -> {args['block_id']}")
                        self.active_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": block_data
                        })
                continue

            self.active_messages.append({"role": "assistant", "content": msg.content})
            return msg.content