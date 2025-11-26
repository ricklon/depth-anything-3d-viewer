# Rebranding Update: vda -> da3d

## Changes Implemented

### 1. CLI Help Text (`da3d/cli/commands.py`)
- Updated the `epilog` examples in the main CLI entry point to use `da3d` instead of `uv run vda`.
- This ensures that users running `da3d --help` see the correct command usage.

### 2. Documentation (`docs/guides/*.md`)
- Updated all guide files to reference `da3d` instead of `uv run vda`.
- Files updated:
    - `docs/guides/static_viewing.md`
    - `docs/guides/screen_capture.md`
    - `docs/guides/realtime_3d.md`
    - `docs/guides/depth_tuning.md`
    - `docs/guides/depth_defaults.md`

### 3. Verification
- Verified `README.md` already uses `da3d` or `uv run da3d`.
- Verified `pyproject.toml` correctly registers the `da3d` script.
- Verified `da3d status` output is correct.
- Verified `da3d --help` output is correct.

## Usage
Users can now consistently use `da3d` (if installed in path) or `uv run da3d` across all commands.

Example:
```bash
da3d webcam3d
```
or
```bash
uv run da3d webcam3d
```
