# Contributing to E-Scooter Safety Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a positive environment
- Report unacceptable behavior to project maintainers

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/YOUR-USERNAME/BEAM-detection-system.git
   cd BEAM-detection-system
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/Methasit-Pun/BEAM-detection-system.git
   ```

## Development Setup

### Prerequisites
- Python 3.8+
- NVIDIA Jetson Nano (for testing on target hardware)
- Compatible camera (CSI or USB)

### Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### Project Structure
```
src/
├── detector.py          # Main detection class
├── model_loader.py      # Model loading and inference
├── video_handler.py     # Video capture and processing
├── violation_checker.py # Violation detection logic
├── alert_system.py      # Alert management
└── utils.py             # Helper functions
```

## How to Contribute

### Reporting Bugs
- Use GitHub Issues
- Include system information (Python version, hardware)
- Provide steps to reproduce
- Include error messages and logs
- Add screenshots if applicable

### Suggesting Enhancements
- Open a GitHub Issue with [FEATURE] tag
- Describe the feature and use case
- Explain expected behavior
- Consider implementation details

### Code Contributions

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Follow coding standards
   - Add tests for new features
   - Update documentation

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```
   
   Commit message format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code formatting
   - `refactor:` Code restructuring
   - `test:` Adding tests
   - `chore:` Maintenance tasks

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

## Coding Standards

### Python Style Guide
- Follow **PEP 8** style guide
- Use **type hints** where appropriate
- Write **docstrings** for all functions and classes
- Maximum line length: 100 characters

### Code Formatting
```bash
# Format code with black
black src/ --line-length 100

# Check with flake8
flake8 src/ --max-line-length 100

# Type checking with mypy
mypy src/
```

### Documentation
- Document all public functions and classes
- Use Google-style docstrings
- Update README.md for significant changes
- Add inline comments for complex logic

Example docstring:
```python
def detect_violations(detections, threshold=0.5):
    """
    Check detections for multi-rider violations
    
    Args:
        detections: List of detection dictionaries
        threshold: Confidence threshold for filtering
    
    Returns:
        List of violation dictionaries with scooter and rider info
    
    Raises:
        ValueError: If detections format is invalid
    """
    pass
```

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_detector.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests
- Add tests for new features
- Maintain test coverage above 80%
- Use descriptive test names
- Test edge cases and error conditions

Example test:
```python
def test_violation_checker_multiple_riders():
    """Test detection of multiple riders on single scooter"""
    checker = ViolationChecker(config)
    detections = [
        {'class_name': 'scooter', 'bbox': [100, 100, 200, 200], 'confidence': 0.9},
        {'class_name': 'person', 'bbox': [110, 110, 150, 180], 'confidence': 0.85},
        {'class_name': 'person', 'bbox': [150, 110, 190, 180], 'confidence': 0.82}
    ]
    violations = checker.check_violations(detections)
    assert len(violations) == 1
    assert violations[0]['rider_count'] == 2
```

## Pull Request Process

### Before Submitting
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No merge conflicts with main branch

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How was this tested?

## Screenshots (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
```

### Review Process
1. Maintainers review code
2. Address feedback and make changes
3. Re-request review after updates
4. Maintainer merges after approval

## Areas for Contribution

### High Priority
- Improve detection accuracy
- Optimize performance on Jetson
- Add unit tests
- Improve documentation

### Medium Priority
- Add support for more camera types
- Implement data logging and analytics
- Create deployment guides
- Add CI/CD pipeline

### Good First Issues
- Fix typos in documentation
- Add code comments
- Improve error messages
- Create example configurations

## Questions?

- Open a GitHub Discussion
- Check existing issues
- Review documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to campus safety! 🛴✨
