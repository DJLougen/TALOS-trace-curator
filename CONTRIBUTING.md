# Contributing to TALOS Trace Curator

First off, thank you for considering contributing to TALOS! It's people like you that make open source software great.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Provide specific examples if possible
* Describe the behavior you observed and what behavior you expected
* Include relevant Python version, OS, and TALOS version

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you would like to see instead
* Explain why this enhancement would be useful

### Pull Requests

The process described here has several goals:

- Maintain TALOS's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible TALOS
- Enable a sustainable system for maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/TALOS-trace-curator.git
cd TALOS-trace-curator

# Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers

# Install development dependencies
pip install flake8 black isort pytest

# Verify installation
python scripts/trace_processor.py --generate-mock
```

### Coding Standards

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting

Run these tools before submitting your PR:

```bash
# Format code
black scripts/
isort --profile black scripts/

# Check linting
flake8 scripts/ --max-line-length=120 --ignore=E501,W503
```

### Testing

While we don't have extensive test suites yet, please:

1. Test your changes with mock data
2. Ensure existing functionality isn't broken
3. Add tests for new features when possible

```bash
# Test with mock data
python scripts/trace_processor.py --generate-mock

# Test scenario generation
python scripts/trace_processor.py --generate-scenarios 10
```

### Documentation

- Update relevant documentation in the `README.md`
- Add docstrings to new functions and classes
- Include examples where helpful
- Update the CLI help text if adding new flags

## Project Structure

```
TALOS-trace-curator/
├── scripts/                 # Core processing scripts
│   ├── trace_processor.py   # Main pipeline
│   ├── generate_traces.py   # Synthetic trace generation
│   └── hf_dataset_converter.py # HF dataset conversion
├── .github/                # GitHub workflows and templates
├── SKILL.md                # Hermes Agent Skill manifest
├── README.md               # Main documentation
├── HACKATHON_SUBMISSION.md # Hackathon submission details
├── example-usage.md        # Usage examples
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT License
```

## Community

- **GitHub Issues**: Bug reports and feature requests
- **HuggingFace**: [Published datasets](https://huggingface.co/djLougen)
- **Discussions**: GitHub Discussions for general questions

## Questions?

If you have any questions, feel free to:
- Open a GitHub Issue
- Check the existing documentation
- Look at the example usage guide

Thank you for contributing to TALOS! 🎉
