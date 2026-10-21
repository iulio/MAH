"""
Telemetry and trajectory profiling for NVIDIA NeMo Agent Toolkit (nvidia-nat) compatible
Agentic Trajectory Interchange Format (ATIF) export.

Provides:
- TelemetryCollector: Thread-safe per-call metrics recording
- RedundantCallDetector: Duplicate tool call detection
- ATIFExporter: ATIF-compliant trajectory export
- TelemetryContext: Async context manager for automatic collection
- ProfilerData: Aggregated data for UI visualization
"""

import asyncio
import hashlib
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from core.config import get_settings_manager, TelemetrySettings


# ==================== Data Models ====================

@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    role: str = ""  # planner, coder, verifier, etc.
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call."""
    tool_name: str = ""
    args_hash: str = ""
    duration_ms: float = 0.0
    exit_code: Optional[int] = None
    stdout_size: int = 0
    stderr_size: int = 0
    success: bool = True
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RedundantCallWarning:
    """Warning for redundant tool calls."""
    tool_name: str
    args_hash: str
    call_count: int
    first_timestamp: str
    last_timestamp: str
    timestamps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ATIFTrajectory:
    """Agentic Trajectory Interchange Format (ATIF) trajectory record."""
    run_id: str
    timestamp: str
    prompt: str
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    diffs: List[Dict[str, Any]] = field(default_factory=list)
    final_test_exit_code: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize to JSONL line."""
        return json.dumps(asdict(self), ensure_ascii=False)


# ==================== TelemetryCollector ====================

class TelemetryCollector:
    """
    Thread-safe telemetry collector for LLM calls and tool calls.
    
    Records per-call metrics with async-compatible methods.
    """

    def __init__(self, settings: Optional[TelemetrySettings] = None):
        self._settings = settings or get_settings_manager().settings.telemetry
        self._lock = threading.RLock()
        self._llm_calls: List[LLMCallMetrics] = []
        self._tool_calls: List[ToolCallMetrics] = []
        self._run_id = str(uuid.uuid4())
        self._start_time = time.time()
        self._enabled = self._settings.enabled

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def _hash_args(self, args: Dict[str, Any]) -> str:
        """Create a deterministic hash of tool arguments."""
        serialized = json.dumps(args, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    # ---- LLM Call Recording ----

    def record_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        latency_ms: float,
        model: str,
        role: str,
        request_id: Optional[str] = None,
    ) -> LLMCallMetrics:
        """Record an LLM call synchronously."""
        if not self._enabled:
            return LLMCallMetrics()
        
        metrics = LLMCallMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            model=model,
            role=role,
            request_id=request_id or str(uuid.uuid4())[:8],
        )
        with self._lock:
            self._llm_calls.append(metrics)
        return metrics

    async def record_llm_call_async(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        latency_ms: float,
        model: str,
        role: str,
        request_id: Optional[str] = None,
    ) -> LLMCallMetrics:
        """Record an LLM call asynchronously (thread-safe)."""
        return await asyncio.to_thread(
            self.record_llm_call,
            prompt_tokens, completion_tokens, reasoning_tokens,
            latency_ms, model, role, request_id
        )

    # ---- Tool Call Recording ----

    def record_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        duration_ms: float,
        exit_code: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        success: bool = True,
        call_id: Optional[str] = None,
    ) -> ToolCallMetrics:
        """Record a tool call synchronously."""
        if not self._enabled:
            return ToolCallMetrics()
        
        metrics = ToolCallMetrics(
            tool_name=tool_name,
            args_hash=self._hash_args(args),
            duration_ms=duration_ms,
            exit_code=exit_code,
            stdout_size=len(stdout),
            stderr_size=len(stderr),
            success=success,
            call_id=call_id or str(uuid.uuid4())[:8],
        )
        with self._lock:
            self._tool_calls.append(metrics)
        return metrics

    async def record_tool_call_async(
        self,
        tool_name: str,
        args: Dict[str, Any],
        duration_ms: float,
        exit_code: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        success: bool = True,
        call_id: Optional[str] = None,
    ) -> ToolCallMetrics:
        """Record a tool call asynchronously (thread-safe)."""
        return await asyncio.to_thread(
            self.record_tool_call,
            tool_name, args, duration_ms, exit_code, stdout, stderr, success, call_id
        )

    # ---- Data Access ----

    def get_llm_calls(self) -> List[LLMCallMetrics]:
        with self._lock:
            return list(self._llm_calls)

    def get_tool_calls(self) -> List[ToolCallMetrics]:
        with self._lock:
            return list(self._tool_calls)

    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics as serializable dicts."""
        with self._lock:
            return {
                "llm_calls": [m.to_dict() for m in self._llm_calls],
                "tool_calls": [m.to_dict() for m in self._tool_calls],
            }

    def clear(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._llm_calls.clear()
            self._tool_calls.clear()

    def get_duration_ms(self) -> float:
        """Get total collection duration in milliseconds."""
        return (time.time() - self._start_time) * 1000


# ==================== RedundantCallDetector ====================

class RedundantCallDetector:
    """
    Detects duplicate tool calls within a session.
    
    Tracks (tool_name, args_hash) pairs and flags when the same tool
    is called with identical arguments more than the threshold.
    """

    def __init__(self, threshold: int = 2):
        self._threshold = threshold
        self._lock = threading.RLock()
        self._call_counts: Dict[str, int] = {}
        self._call_timestamps: Dict[str, List[str]] = {}
        self._warnings: List[RedundantCallWarning] = []

    @property
    def threshold(self) -> int:
        return self._threshold

    @threshold.setter
    def threshold(self, value: int) -> None:
        if value < 1:
            raise ValueError("Threshold must be >= 1")
        self._threshold = value

    def _make_key(self, tool_name: str, args_hash: str) -> str:
        return f"{tool_name}:{args_hash}"

    def record_call(self, tool_name: str, args_hash: str) -> Optional[RedundantCallWarning]:
        """
        Record a tool call and return a warning if redundant.
        
        Returns None if not redundant, otherwise returns the warning.
        """
        key = self._make_key(tool_name, args_hash)
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        with self._lock:
            count = self._call_counts.get(key, 0) + 1
            self._call_counts[key] = count
            
            if key not in self._call_timestamps:
                self._call_timestamps[key] = []
            self._call_timestamps[key].append(timestamp)
            
            if count == self._threshold:
                # First time hitting threshold - create warning
                warning = RedundantCallWarning(
                    tool_name=tool_name,
                    args_hash=args_hash,
                    call_count=count,
                    first_timestamp=self._call_timestamps[key][0],
                    last_timestamp=timestamp,
                    timestamps=list(self._call_timestamps[key]),
                )
                self._warnings.append(warning)
                return warning
            elif count > self._threshold:
                # Update existing warning
                for w in self._warnings:
                    if w.tool_name == tool_name and w.args_hash == args_hash:
                        w.call_count = count
                        w.last_timestamp = timestamp
                        w.timestamps = list(self._call_timestamps[key])
                        return w
        return None

    async def record_call_async(self, tool_name: str, args_hash: str) -> Optional[RedundantCallWarning]:
        """Record a tool call asynchronously."""
        return await asyncio.to_thread(self.record_call, tool_name, args_hash)

    def get_warnings(self) -> List[RedundantCallWarning]:
        with self._lock:
            return list(self._warnings)

    def get_call_count(self, tool_name: str, args_hash: str) -> int:
        with self._lock:
            return self._call_counts.get(self._make_key(tool_name, args_hash), 0)

    def clear(self) -> None:
        with self._lock:
            self._call_counts.clear()
            self._call_timestamps.clear()
            self._warnings.clear()


# ==================== ATIFExporter ====================

class ATIFExporter:
    """
    Exports agent trajectories in Agentic Trajectory Interchange Format (ATIF).
    
    Saves to ./trajectories/{run_id}.jsonl (one JSONL per run).
    Compatible with NVIDIA NeMo Agent Toolkit.
    """

    def __init__(
        self,
        export_dir: Optional[str] = None,
        settings: Optional[TelemetrySettings] = None,
    ):
        self._settings = settings or get_settings_manager().settings.telemetry
        self._export_dir = Path(export_dir or self._settings.export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def _build_trajectory(
        self,
        run_id: str,
        prompt: str,
        llm_calls: List[LLMCallMetrics],
        tool_calls: List[ToolCallMetrics],
        diffs: List[Dict[str, Any]],
        final_test_exit_code: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ATIFTrajectory:
        """Build ATIF trajectory from collected metrics."""
        
        # Build reasoning trace from LLM calls
        reasoning_trace = []
        for call in llm_calls:
            reasoning_trace.append({
                "role": call.role,
                "content": f"LLM call to {call.model}",
                "tokens": {
                    "prompt": call.prompt_tokens,
                    "completion": call.completion_tokens,
                    "reasoning": call.reasoning_tokens,
                    "total": call.total_tokens,
                },
                "latency_ms": call.latency_ms,
                "model": call.model,
                "timestamp": call.timestamp,
                "request_id": call.request_id,
            })

        # Build tool calls list
        tool_calls_list = []
        for call in tool_calls:
            tool_calls_list.append({
                "name": call.tool_name,
                "args_hash": call.args_hash,
                "result": "success" if call.success else "failure",
                "duration_ms": call.duration_ms,
                "exit_code": call.exit_code,
                "stdout_size": call.stdout_size,
                "stderr_size": call.stderr_size,
                "timestamp": call.timestamp,
                "call_id": call.call_id,
            })

        # Build metadata
        models_used = {}
        for call in llm_calls:
            if call.model not in models_used:
                models_used[call.model] = {"calls": 0, "total_tokens": 0, "total_latency_ms": 0.0}
            models_used[call.model]["calls"] += 1
            models_used[call.model]["total_tokens"] += call.total_tokens
            models_used[call.model]["total_latency_ms"] += call.latency_ms

        meta = {
            "models": models_used,
            "duration_ms": (metadata or {}).get("duration_ms", 0),
            "total_llm_calls": len(llm_calls),
            "total_tool_calls": len(tool_calls),
            "total_tokens": sum(c.total_tokens for c in llm_calls),
        }
        if metadata:
            meta.update(metadata)

        return ATIFTrajectory(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            prompt=prompt,
            reasoning_trace=reasoning_trace,
            tool_calls=tool_calls_list,
            diffs=diffs,
            final_test_exit_code=final_test_exit_code,
            metadata=meta,
        )

    def export(
        self,
        run_id: str,
        prompt: str,
        llm_calls: List[LLMCallMetrics],
        tool_calls: List[ToolCallMetrics],
        diffs: Optional[List[Dict[str, Any]]] = None,
        final_test_exit_code: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export trajectory to JSONL file."""
        trajectory = self._build_trajectory(
            run_id=run_id,
            prompt=prompt,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            diffs=diffs or [],
            final_test_exit_code=final_test_exit_code,
            metadata=metadata,
        )
        
        file_path = self._export_dir / f"{run_id}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(trajectory.to_jsonl() + "\n")
        
        return file_path

    async def export_async(
        self,
        run_id: str,
        prompt: str,
        llm_calls: List[LLMCallMetrics],
        tool_calls: List[ToolCallMetrics],
        diffs: Optional[List[Dict[str, Any]]] = None,
        final_test_exit_code: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export trajectory asynchronously."""
        return await asyncio.to_thread(
            self.export,
            run_id, prompt, llm_calls, tool_calls, diffs, final_test_exit_code, metadata
        )

    def export_from_collector(
        self,
        collector: TelemetryCollector,
        prompt: str,
        diffs: Optional[List[Dict[str, Any]]] = None,
        final_test_exit_code: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export trajectory directly from a TelemetryCollector."""
        return self.export(
            run_id=collector.run_id,
            prompt=prompt,
            llm_calls=collector.get_llm_calls(),
            tool_calls=collector.get_tool_calls(),
            diffs=diffs,
            final_test_exit_code=final_test_exit_code,
            metadata={"duration_ms": collector.get_duration_ms(), **(metadata or {})},
        )

    def list_exports(self) -> List[Path]:
        """List all exported trajectory files."""
        return sorted(self._export_dir.glob("*.jsonl"))

    def load_trajectory(self, run_id: str) -> Optional[ATIFTrajectory]:
        """Load a trajectory from file."""
        file_path = self._export_dir / f"{run_id}.jsonl"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        return ATIFTrajectory(**data)


# ==================== TelemetryContext ====================

class TelemetryContext:
    """
    Async context manager for automatic telemetry collection.
    
    Usage:
        async with TelemetryContext() as tel:
            # LLM calls and tool calls are automatically recorded
            await agent.run(...)
        
        # Access collected metrics
        metrics = tel.get_metrics()
    """

    def __init__(
        self,
        collector: Optional[TelemetryCollector] = None,
        detector: Optional[RedundantCallDetector] = None,
        exporter: Optional[ATIFExporter] = None,
        settings: Optional[TelemetrySettings] = None,
    ):
        self._settings = settings or get_settings_manager().settings.telemetry
        self._collector = collector or TelemetryCollector(self._settings)
        self._detector = detector or RedundantCallDetector(self._settings.redundant_threshold)
        self._exporter = exporter or ATIFExporter(settings=self._settings)
        self._prompt: str = ""
        self._diffs: List[Dict[str, Any]] = []
        self._final_exit_code: int = 0
        self._metadata: Dict[str, Any] = {}

    @property
    def collector(self) -> TelemetryCollector:
        return self._collector

    @property
    def detector(self) -> RedundantCallDetector:
        return self._detector

    @property
    def exporter(self) -> ATIFExporter:
        return self._exporter

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def add_diff(self, file: str, diff: str, status: str) -> None:
        """Add a file diff to the trajectory."""
        self._diffs.append({"file": file, "diff": diff, "status": status})

    def set_final_exit_code(self, code: int) -> None:
        self._final_exit_code = code

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    async def __aenter__(self) -> "TelemetryContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._settings.enabled and self._prompt:
            await self._exporter.export_async(
                run_id=self._collector.run_id,
                prompt=self._prompt,
                llm_calls=self._collector.get_llm_calls(),
                tool_calls=self._collector.get_tool_calls(),
                diffs=self._diffs,
                final_test_exit_code=self._final_exit_code,
                metadata=self._metadata,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "llm_calls": [m.to_dict() for m in self._collector.get_llm_calls()],
            "tool_calls": [m.to_dict() for m in self._collector.get_tool_calls()],
            "redundant_warnings": [w.to_dict() for w in self._detector.get_warnings()],
            "run_id": self._collector.run_id,
            "duration_ms": self._collector.get_duration_ms(),
        }

    def export_trajectory(self) -> Path:
        """Synchronously export the trajectory."""
        return self._exporter.export(
            run_id=self._collector.run_id,
            prompt=self._prompt,
            llm_calls=self._collector.get_llm_calls(),
            tool_calls=self._collector.get_tool_calls(),
            diffs=self._diffs,
            final_test_exit_code=self._final_exit_code,
            metadata=self._metadata,
        )

    async def export_trajectory_async(self) -> Path:
        """Asynchronously export the trajectory."""
        return await self._exporter.export_async(
            run_id=self._collector.run_id,
            prompt=self._prompt,
            llm_calls=self._collector.get_llm_calls(),
            tool_calls=self._collector.get_tool_calls(),
            diffs=self._diffs,
            final_test_exit_code=self._final_exit_code,
            metadata=self._metadata,
        )


# ==================== ProfilerData ====================

@dataclass
class ProfilerData:
    """
    Aggregated profiler data for UI visualization.
    
    Provides token consumption per role, latency waterfall,
    redundant call warnings, and export to dict for Plotly charts.
    """
    
    run_id: str
    duration_ms: float
    llm_calls: List[LLMCallMetrics]
    tool_calls: List[ToolCallMetrics]
    redundant_warnings: List[RedundantCallWarning]

    @classmethod
    def from_collector(
        cls,
        collector: TelemetryCollector,
        detector: RedundantCallDetector,
    ) -> "ProfilerData":
        return cls(
            run_id=collector.run_id,
            duration_ms=collector.get_duration_ms(),
            llm_calls=collector.get_llm_calls(),
            tool_calls=collector.get_tool_calls(),
            redundant_warnings=detector.get_warnings(),
        )

    @classmethod
    def from_context(cls, context: TelemetryContext) -> "ProfilerData":
        return cls.from_collector(context.collector, context.detector)

    # ---- Token Consumption ----

    def tokens_by_role(self) -> Dict[str, Dict[str, int]]:
        """Token consumption aggregated by agent role."""
        result: Dict[str, Dict[str, int]] = {}
        for call in self.llm_calls:
            role = call.role or "unknown"
            if role not in result:
                result[role] = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
            result[role]["prompt"] += call.prompt_tokens
            result[role]["completion"] += call.completion_tokens
            result[role]["reasoning"] += call.reasoning_tokens
            result[role]["total"] += call.total_tokens
        return result

    def tokens_by_model(self) -> Dict[str, Dict[str, int]]:
        """Token consumption aggregated by model."""
        result: Dict[str, Dict[str, int]] = {}
        for call in self.llm_calls:
            model = call.model or "unknown"
            if model not in result:
                result[model] = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
            result[model]["prompt"] += call.prompt_tokens
            result[model]["completion"] += call.completion_tokens
            result[model]["reasoning"] += call.reasoning_tokens
            result[model]["total"] += call.total_tokens
        return result

    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    # ---- Latency Waterfall ----

    def latency_waterfall(self) -> List[Dict[str, Any]]:
        """Latency waterfall data for visualization."""
        events = []
        current_time = 0.0
        
        for call in self.llm_calls:
            events.append({
                "type": "llm",
                "role": call.role,
                "model": call.model,
                "start_ms": current_time,
                "duration_ms": call.latency_ms,
                "tokens": call.total_tokens,
                "request_id": call.request_id,
            })
            current_time += call.latency_ms
        
        for call in self.tool_calls:
            events.append({
                "type": "tool",
                "name": call.tool_name,
                "start_ms": current_time,
                "duration_ms": call.duration_ms,
                "success": call.success,
                "exit_code": call.exit_code,
                "call_id": call.call_id,
            })
            current_time += call.duration_ms
        
        return events

    # ---- Redundant Calls ----

    def redundant_call_summary(self) -> List[Dict[str, Any]]:
        """Summary of redundant calls for UI."""
        return [w.to_dict() for w in self.redundant_warnings]

    # ---- Export for Plotly ----

    def to_plotly_data(self) -> Dict[str, Any]:
        """Export all data in a format suitable for Plotly charts."""
        return {
            "run_id": self.run_id,
            "duration_ms": self.duration_ms,
            "tokens_by_role": self.tokens_by_role(),
            "tokens_by_model": self.tokens_by_model(),
            "total_tokens": self.total_tokens(),
            "latency_waterfall": self.latency_waterfall(),
            "redundant_calls": self.redundant_call_summary(),
            "llm_call_count": len(self.llm_calls),
            "tool_call_count": len(self.tool_calls),
            "success_rate": (
                sum(1 for c in self.tool_calls if c.success) / len(self.tool_calls)
                if self.tool_calls else 1.0
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization to dict."""
        return {
            "run_id": self.run_id,
            "duration_ms": self.duration_ms,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "redundant_warnings": [w.to_dict() for w in self.redundant_warnings],
            "plotly_data": self.to_plotly_data(),
        }


# ==================== Integration Helpers ====================

def create_telemetry_context(
    prompt: str = "",
    settings: Optional[TelemetrySettings] = None,
) -> TelemetryContext:
    """Factory function to create a configured TelemetryContext."""
    ctx = TelemetryContext(settings=settings)
    if prompt:
        ctx.set_prompt(prompt)
    return ctx


async def record_llm_call_in_harness(
    harness: "AgentHarness",
    role: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    latency_ms: float,
) -> None:
    """
    Helper to record LLM call metrics from AgentHarness.
    
    This integrates with AgentHarness._stream_chat_completion.
    """
    # This would be called from within the harness after a completion
    # The harness would need to hold a reference to a TelemetryCollector
    pass


async def record_tool_call_in_harness(
    harness: "AgentHarness",
    tool_name: str,
    args: Dict[str, Any],
    duration_ms: float,
    exit_code: Optional[int],
    stdout: str,
    stderr: str,
    success: bool,
) -> Optional[RedundantCallWarning]:
    """
    Helper to record tool call metrics from AgentHarness.
    
    This integrates with AgentHarness._execute_tool.
    Returns redundant call warning if detected.
    """
    # This would be called from within the harness after a tool execution
    # The harness would need to hold references to collector and detector
    pass


# ==================== Module Exports ====================

__all__ = [
    "LLMCallMetrics",
    "ToolCallMetrics",
    "RedundantCallWarning",
    "ATIFTrajectory",
    "TelemetryCollector",
    "RedundantCallDetector",
    "ATIFExporter",
    "TelemetryContext",
    "ProfilerData",
    "create_telemetry_context",
    "record_llm_call_in_harness",
    "record_tool_call_in_harness",
]