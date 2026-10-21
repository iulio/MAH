"""
Git Shadow Checkpoints - Automatic Git-based rollback system for agent runs.

A checkpoint is a snapshot commit of the *entire* working tree (tracked + untracked,
respecting .gitignore) tagged `harness-checkpoint-<timestamp>`. Snapshots are built with
a temporary index, so creating one never modifies the working tree, the real index,
HEAD or the current branch.

Provides:
- Automatic checkpoint creation before agent loops
- 1-click rollback of every file touched by an agent run (edits, creations, deletions)
- Commit accepted changes with a message
- Checkpoint listing, diffing and cleanup (checkpoints survive app restarts as git tags)
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CHECKPOINT_PREFIX = "harness-checkpoint-"
DEFAULT_EXCLUDES = [".venv/", "venv/", "node_modules/", "__pycache__/", "*.pyc"]
FALLBACK_IDENTITY = ("NIM Coding Assistant", "nim-assistant@local")


@dataclass
class CheckpointInfo:
    """Information about a git checkpoint."""
    checkpoint_id: str
    timestamp: datetime
    message: str
    branch_name: str  # Kept for UI compatibility; equals the tag name
    stash_ref: Optional[str] = None  # Unused: snapshots are tagged commits, not stashes
    is_active: bool = False
    files_changed: List[str] = field(default_factory=list)
    commit_hash: Optional[str] = None  # Hash of the accepted commit, if any
    snapshot_commit: Optional[str] = None


class GitGuard:
    """Manages Git shadow checkpoints for automatic rollback of agent runs."""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()
        self._checkpoints: Dict[str, CheckpointInfo] = {}
        self._current_checkpoint: Optional[CheckpointInfo] = None
        self.available = self._ensure_git_repo()
        if self.available:
            self._load_existing_checkpoints()

    # ==================== Git plumbing ====================

    def _run_git(
        self,
        args: List[str],
        capture: bool = True,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        """Run a git command in the workspace."""
        full_env = {**os.environ, **env} if env else None
        return subprocess.run(
            ["git"] + args,
            cwd=self.workspace_dir,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=full_env,
        )

    def _ensure_git_repo(self) -> bool:
        """Ensure the workspace is the root of a git repository, initializing one if needed."""
        try:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            top = self._run_git(["rev-parse", "--show-toplevel"])
            toplevel = Path(top.stdout.strip()).resolve() if top.returncode == 0 and top.stdout.strip() else None
            if toplevel is None or toplevel != self.workspace_dir:
                # Not a repo, or nested inside a parent repo we must not touch.
                init = self._run_git(["init"])
                if init.returncode != 0:
                    print(f"[GitGuard] Failed to init git repo: {init.stderr}")
                    return False
            self._write_default_excludes()
            return True
        except Exception as e:
            print(f"[GitGuard] Git unavailable: {e}")
            return False

    def _git_dir(self) -> Path:
        result = self._run_git(["rev-parse", "--absolute-git-dir"])
        return Path(result.stdout.strip())

    def _write_default_excludes(self) -> None:
        """Keep heavy directories out of snapshots via the repo-local .git/info/exclude."""
        exclude = self._git_dir() / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        missing = [p for p in DEFAULT_EXCLUDES if p not in existing.splitlines()]
        if missing:
            with open(exclude, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write("\n".join(missing) + "\n")

    def _identity_env(self) -> Dict[str, str]:
        """Author/committer fallback so commits work on machines without a git identity."""
        name = self._run_git(["config", "user.name"]).stdout.strip() or FALLBACK_IDENTITY[0]
        email = self._run_git(["config", "user.email"]).stdout.strip() or FALLBACK_IDENTITY[1]
        return {
            "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email,
        }

    def _has_head(self) -> bool:
        return self._run_git(["rev-parse", "--verify", "-q", "HEAD"]).returncode == 0

    def _get_current_branch(self) -> str:
        result = self._run_git(["branch", "--show-current"])
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "HEAD"

    def _get_status_files(self) -> List[str]:
        """List changed files in the working directory (including untracked)."""
        result = self._run_git(["status", "--porcelain", "-uall"])
        if result.returncode != 0:
            return []
        files = []
        for line in result.stdout.splitlines():
            if len(line) > 3:
                path = line[3:]
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                files.append(path.strip('"'))
        return files

    def _snapshot_tree(self) -> str:
        """Write the whole working tree into a tree object using a throwaway index."""
        tmp_dir = tempfile.mkdtemp(prefix="harness-index-")
        env = {"GIT_INDEX_FILE": os.path.join(tmp_dir, "index")}
        try:
            add = self._run_git(["add", "-A"], env=env)
            if add.returncode != 0:
                raise RuntimeError(f"git add failed: {add.stderr.strip()}")
            tree = self._run_git(["write-tree"], env=env)
            if tree.returncode != 0:
                raise RuntimeError(f"git write-tree failed: {tree.stderr.strip()}")
            return tree.stdout.strip()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _snapshot_files(self, commit: str) -> List[str]:
        result = self._run_git(["ls-tree", "-r", "-z", "--name-only", commit])
        if result.returncode != 0:
            raise RuntimeError(f"Cannot read snapshot {commit}: {result.stderr.strip()}")
        return [p for p in result.stdout.split("\0") if p]

    def _current_files(self) -> List[str]:
        """Tracked + untracked (non-ignored) files that currently exist on disk."""
        result = self._run_git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        files = [p for p in result.stdout.split("\0") if p]
        return [p for p in files if (self.workspace_dir / p).is_file()]

    def _safe_path(self, rel: str) -> Optional[Path]:
        target = (self.workspace_dir / rel).resolve()
        try:
            target.relative_to(self.workspace_dir)
        except ValueError:
            return None
        return target

    def _load_existing_checkpoints(self) -> None:
        """Rebuild checkpoint metadata from tags so they survive restarts and Streamlit reruns."""
        result = self._run_git(
            ["for-each-ref", f"refs/tags/{CHECKPOINT_PREFIX}*",
             "--format=%(refname:short)|%(objectname)|%(creatordate:iso-strict)|%(contents:subject)"]
        )
        if result.returncode != 0:
            return
        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) < 4 or parts[0] in self._checkpoints:
                continue
            name, commit, date, subject = parts
            try:
                ts = datetime.fromisoformat(date).replace(tzinfo=None)
            except ValueError:
                ts = datetime.now()
            self._checkpoints[name] = CheckpointInfo(
                checkpoint_id=name, timestamp=ts, message=subject, branch_name=name,
                is_active=False, snapshot_commit=commit,
            )

    # ==================== Public API ====================

    def create_checkpoint(self, message: str = "Pre-agent checkpoint") -> CheckpointInfo:
        """
        Snapshot the current working tree as a tagged commit before an agent run.

        The working tree is left untouched. Raises RuntimeError if git is unavailable.
        """
        if not self.available:
            raise RuntimeError("Git is not available; cannot create checkpoint")

        timestamp = datetime.now()
        base_id = f"{CHECKPOINT_PREFIX}{timestamp.strftime('%Y%m%d-%H%M%S')}"
        checkpoint_id, n = base_id, 1
        while checkpoint_id in self._checkpoints or self._run_git(
            ["rev-parse", "--verify", "-q", f"refs/tags/{checkpoint_id}"]
        ).returncode == 0:
            n += 1
            checkpoint_id = f"{base_id}-{n}"

        files_changed = self._get_status_files()
        tree = self._snapshot_tree()

        commit_args = ["commit-tree", tree, "-m", message]
        if self._has_head():
            commit_args += ["-p", "HEAD"]
        commit = self._run_git(commit_args, env=self._identity_env())
        if commit.returncode != 0:
            raise RuntimeError(f"Failed to create snapshot commit: {commit.stderr.strip()}")
        commit_hash = commit.stdout.strip()

        tag = self._run_git(["tag", checkpoint_id, commit_hash])
        if tag.returncode != 0:
            raise RuntimeError(f"Failed to tag checkpoint: {tag.stderr.strip()}")

        checkpoint = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            message=message,
            branch_name=checkpoint_id,
            is_active=True,
            files_changed=files_changed,
            snapshot_commit=commit_hash,
        )
        self._checkpoints[checkpoint_id] = checkpoint
        self._current_checkpoint = checkpoint
        return checkpoint

    def rollback_to_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore the workspace to exactly the state captured by a checkpoint:
        modified files are restored, deleted files come back, and files created
        after the checkpoint are removed.
        """
        checkpoint = self._find_checkpoint(checkpoint_id)
        if not checkpoint or not checkpoint.snapshot_commit:
            return {"success": False, "error": f"Checkpoint not found: {checkpoint_id}"}

        commit = checkpoint.snapshot_commit
        try:
            snapshot_files = set(self._snapshot_files(commit))
            removed: List[str] = []
            for rel in self._current_files():
                if rel in snapshot_files:
                    continue
                target = self._safe_path(rel)
                if target and target.is_file():
                    target.unlink()
                    removed.append(rel)
                    self._prune_empty_dirs(target.parent)

            tmp_dir = tempfile.mkdtemp(prefix="harness-index-")
            env = {"GIT_INDEX_FILE": os.path.join(tmp_dir, "index")}
            try:
                read = self._run_git(["read-tree", commit], env=env)
                if read.returncode != 0:
                    return {"success": False, "error": f"Failed to read snapshot: {read.stderr.strip()}"}
                checkout = self._run_git(["checkout-index", "-a", "-f"], env=env)
                if checkout.returncode != 0:
                    return {"success": False, "error": f"Failed to restore files: {checkout.stderr.strip()}"}
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            checkpoint.is_active = False
            if self._current_checkpoint is checkpoint:
                self._current_checkpoint = None

            return {
                "success": True,
                "checkpoint_id": checkpoint_id,
                "message": f"Rolled back to {checkpoint_id}",
                "files_restored": sorted(snapshot_files),
                "files_removed": removed,
            }
        except Exception as e:
            return {"success": False, "error": f"Rollback failed: {e}"}

    def _prune_empty_dirs(self, directory: Path) -> None:
        while directory != self.workspace_dir and directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent

    def rollback_last(self) -> Dict[str, Any]:
        """Rollback to the most recent active checkpoint."""
        active = [c for c in self._checkpoints.values() if c.is_active]
        if not active:
            return {"success": False, "error": "No active checkpoints to rollback to"}
        latest = max(active, key=lambda c: c.timestamp)
        return self.rollback_to_checkpoint(latest.checkpoint_id)

    def commit_accepted_changes(self, message: str) -> Dict[str, Any]:
        """Commit the current working tree on the current branch and close the active checkpoint."""
        try:
            add = self._run_git(["add", "-A"])
            if add.returncode != 0:
                return {"success": False, "error": f"Failed to stage changes: {add.stderr.strip()}"}

            if self._has_head() and self._run_git(["diff", "--cached", "--quiet"]).returncode == 0:
                commit_hash = self._run_git(["rev-parse", "HEAD"]).stdout.strip()
                self._close_active_checkpoint(commit_hash)
                return {"success": True, "commit_hash": commit_hash, "message": "Nothing to commit", "no_changes": True}

            commit = self._run_git(["commit", "-m", message], env=self._identity_env())
            if commit.returncode != 0:
                return {"success": False, "error": f"Failed to commit: {(commit.stderr or commit.stdout).strip()}"}

            commit_hash = self._run_git(["rev-parse", "HEAD"]).stdout.strip() or None
            self._close_active_checkpoint(commit_hash)
            return {"success": True, "commit_hash": commit_hash, "message": message}
        except Exception as e:
            return {"success": False, "error": f"Commit failed: {e}"}

    def _close_active_checkpoint(self, commit_hash: Optional[str]) -> None:
        if self._current_checkpoint:
            self._current_checkpoint.is_active = False
            self._current_checkpoint.commit_hash = commit_hash
            self._current_checkpoint = None

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all checkpoints, most recent first."""
        return sorted(self._checkpoints.values(), key=lambda c: c.timestamp, reverse=True)

    def get_active_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the checkpoint of the latest agent run that has not been accepted or rolled back."""
        return self._current_checkpoint

    def _find_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointInfo]:
        return self._checkpoints.get(checkpoint_id)

    def cleanup_old_checkpoints(self, keep: int = 10) -> int:
        """Delete all but the most recent `keep` checkpoint tags. Returns number removed."""
        ordered = sorted(self._checkpoints.values(), key=lambda c: c.timestamp)
        to_remove = ordered[:-keep] if keep > 0 else ordered
        removed = 0
        for cp in to_remove:
            if cp is self._current_checkpoint:
                continue
            self._run_git(["tag", "-d", cp.checkpoint_id])
            del self._checkpoints[cp.checkpoint_id]
            removed += 1
        return removed

    def get_diff_since_checkpoint(self, checkpoint_id: str) -> str:
        """Unified diff between a checkpoint snapshot and the current working tree."""
        checkpoint = self._find_checkpoint(checkpoint_id)
        if not checkpoint or not checkpoint.snapshot_commit:
            return f"Checkpoint not found: {checkpoint_id}"
        try:
            tree = self._snapshot_tree()
            result = self._run_git(["diff", "--no-color", checkpoint.snapshot_commit, tree])
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Diff failed: {e}"

    def is_clean(self) -> bool:
        """Check if the working directory has no uncommitted changes."""
        result = self._run_git(["status", "--porcelain"])
        return result.returncode == 0 and not result.stdout.strip()

    def get_repo_status(self) -> Dict[str, Any]:
        """Repository status for the UI."""
        status_files = self._get_status_files()
        active_cp = self.get_active_checkpoint()
        return {
            "is_git_repo": self.available,
            "current_branch": self._get_current_branch(),
            "is_clean": len(status_files) == 0,
            "changed_files": status_files,
            "active_checkpoint": active_cp.checkpoint_id if active_cp else None,
            "total_checkpoints": len(self._checkpoints),
            "checkpoints": [
                {
                    "id": cp.checkpoint_id,
                    "timestamp": cp.timestamp.isoformat(),
                    "message": cp.message,
                    "is_active": cp.is_active,
                    "files_changed": len(cp.files_changed),
                    "commit_hash": cp.commit_hash,
                }
                for cp in self.list_checkpoints()
            ],
        }
