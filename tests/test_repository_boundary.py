import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src/romanian_news"


def test_python_imports_use_the_standalone_namespace() -> None:
    forbidden = ("chartly", "apps")
    for path in (*PACKAGE.rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
        assert not any(module.startswith(forbidden) for module in modules), path


def test_news_schema_and_repository_exclude_private_chartly_domains() -> None:
    migration = (PACKAGE / "catalog/migrations/0001_initial.sql").read_text()

    assert "debt_transcript_projection_items" not in migration
    assert not (ROOT / "chartly").exists()
    assert not (ROOT / "apps").exists()
    assert not any(PACKAGE.glob("debt*"))
    assert not any(PACKAGE.glob("budget*"))


def test_repository_uses_mit_and_carries_the_htmx_notice() -> None:
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    assert "Zero-Clause BSD" in (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
