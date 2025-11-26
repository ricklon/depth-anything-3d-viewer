# Project Next Steps & Roadmap

## 1. Codebase Cleanup & Best Practices
The project has grown organically and needs refactoring to be production-ready.

- [x] **Package Structure:**
    - [x] Ensure `da3d` is a proper Python package with clean imports.
    - [x] Move scripts (`verify_scale.py`, `analyze_depth_layers.py`) into the package or a `tools/` directory.
- [x] **Type Hinting & Linting:**
    - [ ] Add comprehensive type hints to all functions.
    - [x] Set up `ruff` or `pylint` for code quality checks.
- [x] **Configuration Management:**
    - [x] Move hardcoded constants (like default focal lengths, model paths) to a config file (YAML/TOML).
- [ ] **Documentation:**
    - Add docstrings to all classes and methods.
    - Update `README.md` with clear installation and usage instructions for the new features.

## 2. Metric Depth Refinement (Ongoing)
- [x] **Layering Fixes:**
    - [x] Implemented bilateral smoothing.
    - [x] Inverted metric depth logic (treating as disparity).
    - [x] Increased input resolution to 1024p for high-quality preset.
- [ ] **Further Tuning:**
    - Continue to tweak SOR and smoothing parameters based on user feedback.

## 3. Testing & Validation
- [x] **Automated Visual Testing:**
    - [x] Created `tests/visual_test.py` and `tests/review_visuals.py`.
    - [x] Integrated Vision Agent for automated feedback.
- [ ] **Unit Tests:**
    - Add unit tests for core logic (projection, mesh generation).

## 4. Features
- [ ] **Recording:**
    - Implement high-quality 3D recording (capturing the rendered view, not just the screen).
- [ ] **Multi-Camera Support:**
    - Robustify camera selection and handling.
