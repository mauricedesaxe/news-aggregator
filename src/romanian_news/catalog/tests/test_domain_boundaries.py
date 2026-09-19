import ast
from pathlib import Path

_D1_NAMES = {"d1_query", "d1_batch"}
_PACKAGE_ROOT = Path(__file__).parents[2]


def test_workflow_domains_do_not_import_d1_transport() -> None:
    assert _d1_imports(_scoped_workflow_files()) == []


def _scoped_workflow_files() -> tuple[Path, ...]:
    analysis_files = tuple(
        path for path in (_PACKAGE_ROOT / "analysis").rglob("*.py") if "tests" not in path.parts
    )
    named_files = tuple(
        _PACKAGE_ROOT / name
        for name in ("daily.py", "feedback.py", "groups.py", "reports.py", "themes.py")
    )
    return (
        *(_PACKAGE_ROOT / "feeds").glob("*.py"),
        *(_PACKAGE_ROOT / "articles").glob("*.py"),
        *(_PACKAGE_ROOT / "youtube").glob("*.py"),
        *analysis_files,
        *named_files,
    )


def _d1_imports(paths: tuple[Path, ...]) -> list[tuple[Path, str]]:
    imports = []
    for path in paths:
        tree = ast.parse(path.read_text())
        imports.extend(_d1_names(path, tree))
    return imports


def _d1_names(path: Path, tree: ast.AST) -> list[tuple[Path, str]]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "romanian_news.storage":
            imports.extend((path, name.name) for name in node.names if name.name in _D1_NAMES)
    return imports
