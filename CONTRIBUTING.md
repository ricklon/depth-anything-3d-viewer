# Contributing to Depth-Anything-3D

Thank you for considering contributing to Depth-Anything-3D! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and professional in all interactions. We're all here to build something useful together.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
- Check the [existing issues](https://github.com/ricklon/depth-anything-3d-viewer/issues) to avoid duplicates
- Use the latest version of the code
- Provide detailed reproduction steps

Include in your bug report:
- Python version and OS
- GPU/CUDA version (if using GPU)
- Complete error message and stack trace
- Minimal code to reproduce the issue
- Expected vs actual behavior

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Check existing issues/PRs for similar suggestions
- Provide clear use cases and examples
- Explain why this would be useful to other users

### Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes:**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation
3. **Test your changes:**
   - Run existing tests
   - Test your changes manually
4. **Commit with clear messages:**
   - Use present tense ("Add feature" not "Added feature")
   - Reference issues when applicable
5. **Submit the PR** with a clear description

## Development Setup

### Quick Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_FORK_USERNAME/depth-anything-3d-viewer
cd depth-anything-3d-viewer

# Install in development mode
pip install -e ".[dev,all]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Project Structure

```
depth-anything-3d-viewer/
├── da3d/                   # Main package
│   ├── viewing/           # 3D mesh viewing (mesh.py)
│   ├── projection/        # 2.5D parallax effects (parallax.py)
│   └── cli/               # Command-line interface
├── docs/                  # Documentation
├── tests/                 # Test suite (coming soon)
└── examples/              # Usage examples (coming soon)
```

### Code Style

We use:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **Type hints** where appropriate (but not mandatory)

```bash
# Format code
black da3d/
isort da3d/

# Check types (optional)
mypy da3d/
```

### Testing

```bash
# Run all tests (when implemented)
pytest

# Run specific test file
pytest tests/test_mesh.py

# Run with coverage
pytest --cov=da3d
```

## Development Guidelines

### Adding New Features

1. **Start with an issue** - Discuss the feature first
2. **Keep it modular** - Follow existing patterns
3. **Add docstrings** - Document functions/classes
4. **Consider performance** - This is a real-time system
5. **Test manually** - Especially for visualization features

### Code Patterns

#### 3D Viewing Features

When adding 3D viewing features, follow the pattern in `da3d/viewing/mesh.py`:
- Support both mesh and point cloud modes
- Use percentile-based depth clamping
- Scale Z-depth proportionally to image dimensions
- Provide subsample options for performance

#### CLI Commands

When adding CLI commands, follow the pattern in `da3d/cli/legacy.py`:
- Use argparse with clear help text
- Provide sensible defaults
- Group related options together
- Add examples in the help text

#### Performance Considerations

- Use `torch.no_grad()` for inference
- Provide subsample options (2-4x)
- Allow resolution limiting
- Support FP16 by default (with FP32 option)
- Avoid unnecessary memory copies

### Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Update README.md for new features
- Add usage examples for CLI commands
- Create guide documents for complex features

## Release Process (for maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag v0.x.x`
4. Push tag: `git push origin v0.x.x`
5. GitHub Actions will handle the release

## Questions?

- Open an [issue](https://github.com/ricklon/depth-anything-3d-viewer/issues) for questions
- Use [Discussions](https://github.com/ricklon/depth-anything-3d-viewer/discussions) for general topics

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
