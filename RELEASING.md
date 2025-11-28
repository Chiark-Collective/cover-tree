# Releasing covertreex

Checklist for publishing a new release to PyPI.

## Prerequisites

- Rust toolchain installed
- Python 3.12 and 3.13 available
- maturin and twine installed
- PyPI token in `.pypi_token.env` (format: `PYPI_API_TOKEN=pypi-...`)

## Release Checklist

1. **Version bump**
   - Update `pyproject.toml` version field.
   - Move `[Unreleased]` section in `CHANGELOG.md` to new version with date.

2. **Commit and push**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to X.Y.Z"
   git push origin main
   ```

3. **Build wheels**
   ```bash
   # Python 3.13 (uses current venv interpreter)
   maturin build --release

   # Python 3.12
   maturin build --release -i python3.12
   ```

4. **Run tests**
   ```bash
   pytest
   ```

5. **Upload to PyPI**
   ```bash
   source .pypi_token.env
   TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_TOKEN" twine upload target/wheels/covertreex-X.Y.Z-*.whl
   ```

6. **Verify**
   - Check https://pypi.org/project/covertreex/
   - Test installation: `pip install covertreex==X.Y.Z`

## Supported Platforms

- **Python**: 3.12, 3.13
- **OS**: Linux (manylinux_2_34_x86_64)
- **Architecture**: x86_64 with AVX2 support

## Notes

- The Rust backend is compiled into the wheel via maturin
- Wheels include the `profiles/*.yaml` runtime presets
- Version in `pyproject.toml` is the single source of truth
