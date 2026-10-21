"""
Repository Map Engine - AST-based codebase navigation for large projects.

Builds a compact structural skeleton of the codebase:
- Parses Python files with ast module (classes, methods, functions, type hints, imports)
- Regex extraction for JS/TS/Go/Rust files
- Import dependency graph with graph ranking
- Token-budgeted output for LLM context injection
"""

import ast
import re
import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import fnmatch


@dataclass
class Symbol:
    """Represents a code symbol (class, function, method, constant)."""
    name: str
    type: str  # class, function, method, constant, import
    signature: str
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None


@dataclass
class FileMap:
    """Structural map of a single file."""
    file_path: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    language: str = ""
    size_bytes: int = 0
    error: Optional[str] = None


@dataclass
class RepoMap:
    """Complete repository map with dependency graph."""
    files: Dict[str, FileMap] = field(default_factory=dict)
    import_graph: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_import_graph: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    symbol_index: Dict[str, List[Symbol]] = field(default_factory=lambda: defaultdict(list))
    total_tokens: int = 0


class PythonASTVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols from Python files."""
    
    def __init__(self, file_path: str, source_lines: List[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.symbols: List[Symbol] = []
        self.imports: List[str] = []
        self.current_class: Optional[str] = None
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}" if module else alias.name)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        old_class = self.current_class
        self.current_class = node.name
        
        # Get class signature with bases
        bases = [base.id if isinstance(base, ast.Name) else ast.unparse(base) for base in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        
        docstring = ast.get_docstring(node)
        
        self.symbols.append(Symbol(
            name=node.name,
            type="class",
            signature=signature,
            file_path=self.file_path,
            line_number=node.lineno,
            docstring=docstring
        ))
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node, is_async=False)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node, is_async=True)
    
    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool):
        # Build signature with type hints
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Handle defaults
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            num_args = len(args)
            for i in range(num_defaults):
                idx = num_args - num_defaults + i
                if idx < len(args):
                    args[idx] += f" = {ast.unparse(defaults[i])}"
        
        # Return type
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        
        async_prefix = "async " if is_async else ""
        signature = f"{async_prefix}def {node.name}({', '.join(args)}){returns}"
        
        docstring = ast.get_docstring(node)
        
        symbol_type = "method" if self.current_class else "function"
        
        self.symbols.append(Symbol(
            name=node.name,
            type=symbol_type,
            signature=signature,
            file_path=self.file_path,
            line_number=node.lineno,
            docstring=docstring,
            parent_class=self.current_class
        ))
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        # Extract module-level constants (UPPER_CASE names)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                value_str = ""
                if isinstance(node.value, ast.Constant):
                    value_str = f" = {repr(node.value.value)}"
                elif isinstance(node.value, (ast.List, ast.Dict, ast.Tuple, ast.Set)):
                    value_str = f" = {ast.unparse(node.value)}"
                
                self.symbols.append(Symbol(
                    name=target.id,
                    type="constant",
                    signature=f"{target.id}{value_str}",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    parent_class=self.current_class
                ))
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        # Annotated assignments (typed constants)
        if isinstance(node.target, ast.Name) and node.target.id.isupper():
            annotation = ast.unparse(node.annotation) if node.annotation else ""
            value_str = f" = {ast.unparse(node.value)}" if node.value else ""
            self.symbols.append(Symbol(
                name=node.target.id,
                type="constant",
                signature=f"{node.target.id}: {annotation}{value_str}",
                file_path=self.file_path,
                line_number=node.lineno,
                parent_class=self.current_class
            ))
        self.generic_visit(node)


class RegexSymbolExtractor:
    """Regex-based symbol extraction for non-Python languages."""
    
    PATTERNS = {
        ".js": {
            "function": r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)",
            "arrow_function": r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            "class": r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?",
            "import": r"import\s+(?:[^'\"\)]+from\s+)?['\"]([^'\"]+)['\"]",
            "export": r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)",
        },
        ".ts": {
            "function": r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)(?:\s*:\s*\w+)?",
            "arrow_function": r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w+)?\s*=>",
            "class": r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+)?",
            "interface": r"(?:export\s+)?interface\s+(\w+)",
            "type": r"(?:export\s+)?type\s+(\w+)\s*=",
            "import": r"import\s+(?:[^'\"\)]+from\s+)?['\"]([^'\"]+)['\"]",
        },
        ".go": {
            "function": r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\([^)]*\)",
            "struct": r"type\s+(\w+)\s+struct\s*{",
            "interface": r"type\s+(\w+)\s+interface\s*{",
            "import": r'import\s+(?:"([^"]+)"|`([^`]+)`)',
        },
        ".rs": {
            "function": r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\([^)]*\)",
            "struct": r"(?:pub\s+)?struct\s+(\w+)",
            "enum": r"(?:pub\s+)?enum\s+(\w+)",
            "trait": r"(?:pub\s+)?trait\s+(\w+)",
            "impl": r"impl(?:\s+<\w+>)?\s+(\w+)",
            "import": r"use\s+([^;]+);",
        },
    }
    
    def __init__(self):
        self.compiled_patterns: Dict[str, Dict[str, re.Pattern]] = {}
        for ext, patterns in self.PATTERNS.items():
            self.compiled_patterns[ext] = {
                name: re.compile(pattern, re.MULTILINE)
                for name, pattern in patterns.items()
            }
    
    def extract(self, file_path: str, content: str) -> Tuple[List[Symbol], List[str]]:
        """Extract symbols and imports from file content."""
        ext = Path(file_path).suffix
        if ext not in self.compiled_patterns:
            return [], []
        
        patterns = self.compiled_patterns[ext]
        symbols = []
        imports = []
        lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            # Check imports
            if "import" in patterns:
                for match in patterns["import"].finditer(line):
                    imports.append(match.group(1))
            
            # Check other symbols
            for sym_type, pattern in patterns.items():
                if sym_type == "import":
                    continue
                for match in pattern.finditer(line):
                    name = match.group(1)
                    signature = line.strip()
                    symbols.append(Symbol(
                        name=name,
                        type=sym_type,
                        signature=signature,
                        file_path=file_path,
                        line_number=line_num
                    ))
        
        return symbols, imports


class RepoMapper:
    """
    Builds a repository map for codebase navigation.
    
    Features:
    - Multi-language support (Python via AST, JS/TS/Go/Rust via regex)
    - Import dependency graph construction
    - Graph ranking (PageRank-style) for relevance scoring
    - Token-budgeted output formatting
    - Keyword-based filtering for task-specific relevance
    """
    
    def __init__(
        self,
        workspace_dir: str,
        token_budget: int = 2000,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        max_file_size_kb: int = 100
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.token_budget = token_budget
        self.include_patterns = include_patterns or [
            "**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"
        ]
        self.exclude_patterns = exclude_patterns or [
            "**/node_modules/**", "**/.git/**", "**/__pycache__/**",
            "**/venv/**", "**/.venv/**", "**/dist/**", "**/build/**",
            "**/*.pyc", "**/*.min.js", "**/*.map"
        ]
        self.max_file_size = max_file_size_kb * 1024
        self._python_visitor = PythonASTVisitor
        self._regex_extractor = RegexSymbolExtractor()
        self._repo_map: Optional[RepoMap] = None
    
    def _match(self, rel_str: str, pattern: str) -> bool:
        """fnmatch that lets `**/x` also match root-level files."""
        return fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch("x/" + rel_str, pattern)

    def _excluded_dir_names(self) -> Set[str]:
        """Directory names pruned during the walk, derived from `**/NAME/**` exclude patterns."""
        names = {".git", ".venv", "venv", "node_modules", "__pycache__"}
        for pattern in self.exclude_patterns:
            m = re.fullmatch(r"\*\*/([^/*]+)/\*\*", pattern)
            if m:
                names.add(m.group(1))
        return names

    def _should_include(self, file_path: Path) -> bool:
        """Check if file should be included in repo map."""
        rel_str = str(file_path.relative_to(self.workspace_dir)).replace("\\", "/")
        if any(self._match(rel_str, p) for p in self.exclude_patterns):
            return False
        return any(self._match(rel_str, p) for p in self.include_patterns)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4

    def _iter_candidate_files(self):
        """Walk the workspace, pruning excluded directories before descending."""
        pruned = self._excluded_dir_names()
        for root, dirs, files in os.walk(self.workspace_dir):
            dirs[:] = sorted(d for d in dirs if d not in pruned)
            for name in sorted(files):
                yield Path(root) / name

    @staticmethod
    def _python_module_name(rel_path: str) -> str:
        parts = rel_path[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _resolve_imports(
        self, file_path: str, imports: List[str], language: str,
        module_map: Dict[str, str], all_files: Set[str]
    ) -> Set[str]:
        """Resolve raw import strings to workspace file paths (internal dependencies only)."""
        resolved: Set[str] = set()
        base_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else ""

        for imp in imports:
            if language == "py":
                candidates = [imp]
                if base_dir:
                    candidates.insert(0, base_dir.replace("/", ".") + "." + imp)
                for cand in candidates:
                    parts = cand.split(".")
                    for i in range(len(parts), 0, -1):
                        target = module_map.get(".".join(parts[:i]))
                        if target:
                            if target != file_path:
                                resolved.add(target)
                            break
                    else:
                        continue
                    break
            elif language in ("js", "ts") and imp.startswith("."):
                joined = os.path.normpath(os.path.join(base_dir, imp)).replace("\\", "/")
                for suffix in ("", ".ts", ".js", "/index.ts", "/index.js"):
                    if joined + suffix in all_files:
                        resolved.add(joined + suffix)
                        break
        return resolved
    
    def scan(self, force_refresh: bool = False) -> RepoMap:
        """
        Scan workspace and build repository map.
        
        Args:
            force_refresh: Force re-scan even if cached
            
        Returns:
            RepoMap with all file maps and dependency graph
        """
        if self._repo_map is not None and not force_refresh:
            return self._repo_map
        
        repo_map = RepoMap()
        
        for file_path in self._iter_candidate_files():
            if not self._should_include(file_path):
                continue
            try:
                if file_path.stat().st_size > self.max_file_size:
                    continue
            except OSError:
                continue
            
            file_map = self._process_file(file_path)
            if file_map and not file_map.error:
                repo_map.files[file_map.file_path] = file_map
                for symbol in file_map.symbols:
                    repo_map.symbol_index[symbol.name].append(symbol)

        # Resolve imports to files so that centrality ranking operates on real file nodes
        all_files = set(repo_map.files)
        module_map = {
            self._python_module_name(p): p for p in all_files if p.endswith(".py")
        }
        for path, file_map in repo_map.files.items():
            for dep in self._resolve_imports(path, file_map.imports, file_map.language, module_map, all_files):
                repo_map.import_graph[path].add(dep)
                repo_map.reverse_import_graph[dep].add(path)
        
        self._repo_map = repo_map
        return repo_map
    
    def _process_file(self, file_path: Path) -> Optional[FileMap]:
        """Process a single file and extract its map."""
        try:
            rel_path = str(file_path.relative_to(self.workspace_dir)).replace("\\", "/")
            content = file_path.read_text(encoding="utf-8")
            size = len(content.encode("utf-8"))
            
            ext = file_path.suffix
            language = ext[1:] if ext else "unknown"
            
            if ext == ".py":
                return self._process_python_file(rel_path, content, language, size)
            else:
                return self._process_regex_file(rel_path, content, language, size)
                
        except Exception as e:
            return FileMap(
                file_path=str(file_path.relative_to(self.workspace_dir)).replace("\\", "/"),
                error=str(e)
            )
    
    def _process_python_file(
        self, rel_path: str, content: str, language: str, size: int
    ) -> FileMap:
        """Process Python file with AST."""
        try:
            tree = ast.parse(content, filename=rel_path)
            source_lines = content.splitlines()
            visitor = self._python_visitor(rel_path, source_lines)
            visitor.visit(tree)
            
            return FileMap(
                file_path=rel_path,
                symbols=visitor.symbols,
                imports=visitor.imports,
                language=language,
                size_bytes=size
            )
        except SyntaxError as e:
            return FileMap(
                file_path=rel_path,
                language=language,
                size_bytes=size,
                error=f"Syntax error: {e.msg} at line {e.lineno}"
            )
        except Exception as e:
            return FileMap(
                file_path=rel_path,
                language=language,
                size_bytes=size,
                error=str(e)
            )
    
    def _process_regex_file(
        self, rel_path: str, content: str, language: str, size: int
    ) -> FileMap:
        """Process non-Python file with regex."""
        symbols, imports = self._regex_extractor.extract(rel_path, content)
        
        return FileMap(
            file_path=rel_path,
            symbols=symbols,
            imports=imports,
            language=language,
            size_bytes=size
        )
    
    def rank_files(self, query: str = "", top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Rank files by relevance using graph centrality and keyword matching.
        
        Args:
            query: User query/keywords for relevance scoring
            top_k: Number of top files to return
            
        Returns:
            List of (file_path, score) tuples sorted by relevance
        """
        if not self._repo_map:
            self.scan()
        
        scores: Dict[str, float] = {path: 0.0 for path in self._repo_map.files}
        query_terms = {t for t in re.split(r"\W+", query.lower()) if len(t) > 1} if query else set()
        
        # 1. Graph centrality: files imported by many others are more important
        for file_path, importers in self._repo_map.reverse_import_graph.items():
            if file_path in self._repo_map.files:
                scores[file_path] += len(importers) * 0.5
        
        # 2. Keyword matching in file paths and symbols
        if query_terms:
            for file_path, file_map in self._repo_map.files.items():
                # Path matching
                path_lower = file_path.lower()
                for term in query_terms:
                    if term in path_lower:
                        scores[file_path] += 2.0
                
                # Symbol matching
                for symbol in file_map.symbols:
                    sym_lower = symbol.name.lower()
                    for term in query_terms:
                        if term in sym_lower:
                            scores[file_path] += 1.0
                            break
        
        # 3. File size penalty (prefer smaller, focused files)
        for file_path, file_map in self._repo_map.files.items():
            size_kb = file_map.size_bytes / 1024
            if size_kb > 50:
                scores[file_path] -= min(size_kb / 100, 2.0)
        
        # Sort and return top-k
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]
    
    def format_repo_map(
        self,
        query: str = "",
        max_tokens: Optional[int] = None,
        include_symbols: bool = True
    ) -> str:
        """
        Format repository map as a concise text block for LLM context.
        
        Args:
            query: Optional query for relevance ranking
            max_tokens: Token budget (uses instance default if None)
            include_symbols: Whether to include symbol details
            
        Returns:
            Formatted repo map string
        """
        if not self._repo_map:
            self.scan()
        
        budget = max_tokens or self.token_budget
        ranked_files = self.rank_files(query, top_k=500)
        
        lines = ["<repo_map>"]
        current_tokens = self._estimate_tokens("\n".join(lines))
        
        for file_path, score in ranked_files:
            if current_tokens >= budget:
                lines.append(f"... (truncated, {len(ranked_files)} files total)")
                break
            
            file_map = self._repo_map.files[file_path]
            
            # File header
            file_header = f"\n{file_path} ({file_map.language})"
            if file_map.error:
                file_header += f" [ERROR: {file_map.error}]"
            
            header_tokens = self._estimate_tokens(file_header)
            if current_tokens + header_tokens >= budget:
                break
            
            lines.append(file_header)
            current_tokens += header_tokens
            
            # Imports
            if file_map.imports:
                imports_line = f"  imports: {', '.join(file_map.imports[:10])}"
                if len(file_map.imports) > 10:
                    imports_line += f" ... (+{len(file_map.imports) - 10} more)"
                imp_tokens = self._estimate_tokens(imports_line)
                if current_tokens + imp_tokens < budget:
                    lines.append(imports_line)
                    current_tokens += imp_tokens
            
            # Symbols
            if include_symbols and file_map.symbols:
                for symbol in file_map.symbols[:15]:  # Limit symbols per file
                    sym_line = f"  {symbol.type}: {symbol.signature}"
                    if symbol.docstring:
                        # Truncate docstring
                        doc = symbol.docstring.split('\n')[0][:80]
                        sym_line += f"  # {doc}"
                    
                    sym_tokens = self._estimate_tokens(sym_line)
                    if current_tokens + sym_tokens >= budget:
                        lines.append("  ...")
                        break
                    lines.append(sym_line)
                    current_tokens += sym_tokens
                
                if len(file_map.symbols) > 15:
                    lines.append(f"  ... (+{len(file_map.symbols) - 15} more symbols)")
        
        lines.append("\n</repo_map>")
        result = "\n".join(lines)
        
        # Update token count
        self._repo_map.total_tokens = self._estimate_tokens(result)
        
        return result
    
    def get_file_map(self, file_path: str) -> Optional[FileMap]:
        """Get detailed map for a specific file."""
        if not self._repo_map:
            self.scan()
        return self._repo_map.files.get(file_path)
    
    def find_symbol(self, name: str) -> List[Symbol]:
        """Find all occurrences of a symbol across the codebase."""
        if not self._repo_map:
            self.scan()
        return self._repo_map.symbol_index.get(name, [])
    
    def get_dependencies(self, file_path: str) -> Set[str]:
        """Get files that import the given file."""
        if not self._repo_map:
            self.scan()
        return self._repo_map.reverse_import_graph.get(file_path, set())
    
    def get_dependents(self, file_path: str) -> Set[str]:
        """Get files imported by the given file."""
        if not self._repo_map:
            self.scan()
        return self._repo_map.import_graph.get(file_path, set())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get repository map statistics."""
        if not self._repo_map:
            self.scan()
        
        total_symbols = sum(len(fm.symbols) for fm in self._repo_map.files.values())
        total_imports = sum(len(fm.imports) for fm in self._repo_map.files.values())
        
        by_language = defaultdict(int)
        for fm in self._repo_map.files.values():
            by_language[fm.language] += 1
        
        return {
            "total_files": len(self._repo_map.files),
            "total_symbols": total_symbols,
            "total_imports": total_imports,
            "by_language": dict(by_language),
            "token_budget": self.token_budget,
            "estimated_tokens": self._repo_map.total_tokens
        }


def create_repo_mapper(
    workspace_dir: str,
    token_budget: int = 2000,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
    max_file_size_kb: int = 100
) -> RepoMapper:
    """Factory function to create a RepoMapper instance."""
    return RepoMapper(
        workspace_dir=workspace_dir,
        token_budget=token_budget,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_file_size_kb=max_file_size_kb
    )