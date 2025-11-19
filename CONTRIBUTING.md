# Contributing to Inferenco Predictions SDK

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites
- Rust 1.77+ with 2024 edition support
- `rustup`, `cargo`, `rustfmt`, and `clippy`

### Setup
```bash
git clone https://github.com/your-org/Inferenco-predictions-SDK.git
cd Inferenco-predictions-SDK/prediction_sdk
cargo build
cargo test
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes
- Write clean, documented code
- Follow Rust naming conventions
- Add tests for new features
- Update documentation as needed

### 3. Format and Lint
```bash
cargo fmt
cargo clippy
```

### 4. Run Tests
```bash
# All tests
cargo test

# Specific test suites
cargo test --lib          # Unit tests
cargo test --test '*'     # Integration tests
cargo test --doc          # Doc tests
```

### 5. Submit a Pull Request
- Push your branch to GitHub
- Open a PR with a clear description
- Reference any related issues
- Wait for CI checks to pass

## Code Standards

### Style
- Run `cargo fmt` before committing
- Address all `cargo clippy` warnings
- Follow the existing code structure

### Documentation
- Add doc comments (`///`) for public APIs
- Include examples in doc comments where helpful
- Update `DOCUMENTATION.md` for significant changes

### Testing
- Write unit tests for individual functions
- Add integration tests for workflows
- Ensure all tests pass before submitting PR

### Commits
- Use clear, descriptive commit messages
- Reference issues when applicable (e.g., `Fix #123: Description`)
- Keep commits focused and atomic

## What to Contribute

### Bug Reports
Open an issue with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Rust version)

### Feature Requests
Open an issue with:
- Clear description of the feature
- Use case and rationale
- Proposed implementation (if applicable)

### Code Contributions
We welcome:
- Bug fixes
- New forecasting models or indicators
- Performance improvements
- Test coverage improvements
- Documentation improvements

## Pull Request Process

1. **Fork and clone** the repository
2. **Create a branch** from `main`
3. **Make your changes** following code standards
4. **Add tests** for new functionality
5. **Update docs** if needed
6. **Run CI locally**:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo build
   cargo test
   ```
7. **Submit PR** with detailed description
8. **Address review feedback** if requested
9. **Wait for approval** and merge

## CI/CD

GitHub Actions automatically runs:
- `cargo fmt --check` - Format validation
- `cargo clippy -- -D warnings` - Lint checks
- `cargo build` - Build verification
- `cargo test` - Full test suite

PRs must pass all CI checks before merging.

## Architecture Guidelines

### Adding New Forecasting Models
1. Add model logic to `src/analysis.rs`
2. Update `ShortForecastResult` or `LongForecastResult` DTOs if needed
3. Integrate into ensemble in `src/impl.rs`
4. Add tests in `tests/` directory
5. Document in `DOCUMENTATION.md`

### Adding New Technical Indicators
1. Use the `ta` crate when possible
2. Add calculation to `src/analysis.rs`
3. Include in `TechnicalSignals` struct
4. Update integration tests
5. Document the indicator and its interpretation

### Modifying Cache Strategy
1. Update TTL logic in `src/cache.rs`
2. Ensure thread-safety with `Arc`
3. Add tests for new cache behavior
4. Document changes in `DOCUMENTATION.md`

## Questions or Issues?

- **Bugs**: Open an issue with the `bug` label
- **Features**: Open an issue with the `enhancement` label
- **Questions**: Open a discussion or contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers and help them learn
- Follow the project's technical standards
