# ProteusPy Development Guide

## Commands
- `make tests`: Run all tests
- `pytest tests/test_file.py`: Run a specific test
- `make format`: Format code with black
- `make bld`: Build package wheels
- `make docs`: Generate documentation
- `make pkg`: Create conda environment for proteusPy
- `make dev`: Build development environment
- `make install`: Install proteusPy package and build databases
- `make install_dev`: Install development version of proteusPy
- `make clean`: Remove proteusPy environment
- `make devclean`: Remove development environment

## Code Style
- Use Black formatter for consistent style
- Write docstrings in reStructuredText format with `:param name: description` style
- Include `:return:`, `:raises:`, and `:type:` in docstrings when applicable
- Use type annotations throughout
- Follow existing import order: stdlib, third-party, local
- Use custom exceptions from DisulfideExceptions.py
- Use snake_case for variables/functions, PascalCase for classes
- Include logging with the configured logger
- Keep functions focused and under 50 lines when possible

## Testing
- Write pytest tests for new functionality
- Tests should be in the `tests/` directory with `test_` prefix