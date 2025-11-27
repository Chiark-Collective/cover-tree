# Releasing pytest-chronicle

Checklist prior to cutting a release or migrating to the standalone repository.

1. **Version bump**
   - Update `tools/pytest-chronicle/src/pytest_chronicle/__init__.py`.
   - Update `pyproject.toml` version field.

2. **Changelog**
   - Add release notes to the standalone repo (e.g., `CHANGELOG.md`).

3. **Dependency audit**
   - Confirm runtime dependencies (`sqlalchemy`, `sqlmodel`, `alembic`, `aiosqlite`, `asyncpg`) remain accurate.
   - Ensure optional extras (`dev`) cover test tooling.

4. **Packaging sanity checks**
   - Run `uv build` to generate sdist/wheel and verify Alembic assets are present.
   - Inspect `pytest_chronicle/alembic/` contents in the artifacts.

5. **Tests & linters**
   - Execute the lightweight suite: `uv run --python 3.12 --with pytest --with sqlmodel --with sqlalchemy --with aiosqlite --with asyncpg --with alembic python -m pytest tools/pytest-chronicle/tests/unit tools/pytest-chronicle/tests/integration -q`.
   - Optionally exercise `pytest-chronicle run` against a sample project (see journal for examples).

6. **Publishing**
   - Use `uv publish --token <PYPI_TOKEN>` once the package is tagged and the standalone repo CI passes.

7. **Monorepo integration**
   - Update `PYTEST_RESULTS_DRIVER` default once the external dependency is vendored.
   - Remove legacy shims (`tools/test_results/*`) after consumers update import paths.
