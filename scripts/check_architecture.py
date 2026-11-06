from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "src/nailong_classifier/application/features"
FORBIDDEN_ASYNCIO_CALLS = {"create_task", "to_thread", "Lock"}
FORBIDDEN_DIRECT_CALLS = {"send", "update_card"}
FORBIDDEN_IMPORT_PARTS = {"adapters", "bot", "integrations", "persistence"}


@dataclass(frozen=True, order=True)
class Violation:
    path: Path
    line: int
    column: int
    message: str

    def format(self, root: Path = PROJECT_ROOT) -> str:
        try:
            path = self.path.relative_to(root)
        except ValueError:
            path = self.path
        return f"{path}:{self.line}:{self.column}: {self.message}"


def _module_is_forbidden(module: str) -> bool:
    return bool(set(module.split(".")) & FORBIDDEN_IMPORT_PARTS)


class BoundaryVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.aliases: dict[str, str] = {}
        self.violations: list[Violation] = []

    def _add(self, node: ast.AST, message: str) -> None:
        self.violations.append(
            Violation(
                self.path,
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
                message,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            self.aliases[local_name] = alias.name
            if _module_is_forbidden(alias.name):
                self._add(node, f"forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            imported = ".".join(part for part in (module, alias.name) if part)
            self.aliases[alias.asname or alias.name] = imported
            if _module_is_forbidden(imported):
                self._add(node, f"forbidden import: {imported}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._qualified_name(node.func)
        final_name = call_name.rsplit(".", 1)[-1]

        if final_name in FORBIDDEN_DIRECT_CALLS:
            self._add(node, f"direct call to {final_name} is forbidden")
        if call_name.startswith("asyncio.") and final_name in FORBIDDEN_ASYNCIO_CALLS:
            self._add(node, f"{call_name} is forbidden in application features")

        self.generic_visit(node)

    def _qualified_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = self._qualified_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""


def check_architecture(features_dir: Path = FEATURES_DIR) -> list[Violation]:
    features_dir = Path(features_dir)
    if not features_dir.is_dir():
        return [Violation(features_dir, 1, 1, "features directory does not exist")]

    violations: list[Violation] = []
    for path in sorted(features_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            violations.append(
                Violation(
                    path,
                    error.lineno or 1,
                    error.offset or 1,
                    f"cannot parse file: {error.msg}",
                )
            )
            continue

        visitor = BoundaryVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return sorted(violations)


def main() -> int:
    violations = check_architecture()
    if not violations:
        print("Architecture boundaries passed.")
        return 0

    for violation in violations:
        print(violation.format(), file=sys.stderr)
    print(
        f"Architecture boundary check failed with {len(violations)} violation(s).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
