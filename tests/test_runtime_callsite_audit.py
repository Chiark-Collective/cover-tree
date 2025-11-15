import re
from pathlib import Path


def test_runtime_config_call_sites_are_explicit() -> None:
    """Ensure cx_config.runtime_config() is not used outside legacy tests."""

    repo_root = Path(__file__).resolve().parents[1]
    target_dirs = ("covertreex", "cli", "tools", "tests")
    allowlist = {
        "tests/test_config.py",
        "tests/test_runtime_callsite_audit.py",
        "covertreex/runtime/config.py",
    }
    pattern = re.compile(r"(?<![A-Za-z0-9_])runtime_config\s*\(")
    offenders: list[str] = []

    for directory in target_dirs:
        base = repo_root / directory
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            rel_path = path.relative_to(repo_root).as_posix()
            if rel_path in allowlist:
                continue
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(rel_path)

    assert not offenders, (
        "Found unexpected cx_config.runtime_config() call sites outside the "
        f"allowlist: {', '.join(sorted(offenders))}"
    )
