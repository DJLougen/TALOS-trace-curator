# Security Policy

## Supported Versions

Currently supported versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | ✅                 |
| 1.x.x   | ❌ (Legacy)        |

## Reporting a Vulnerability

We take the security of TALOS Trace Curator seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **Do NOT open a public issue** - Security vulnerabilities should be reported privately
2. **Email the maintainer** at djLougen@github.com with:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if you have one)

3. **Include relevant details**:
   - Version of TALOS affected
   - Operating system and Python version
   - Any relevant configuration files (with sensitive data removed)

### What to Expect

- **Acknowledgment**: You'll receive a response within 48 hours
- **Assessment**: We'll assess the severity and impact
- **Timeline**: We'll provide an estimated fix timeline
- **Credit**: You'll be credited in the release notes (unless you prefer anonymity)

### Vulnerability Categories

We prioritize vulnerabilities in this order:

1. **Critical**: Remote code execution, data exfiltration, PII leakage
2. **High**: Authentication bypass, privilege escalation, data corruption
3. **Medium**: Information disclosure, denial of service
4. **Low**: Minor security improvements, best practice violations

### Scope

This security policy applies to:
- TALOS core processing scripts
- Data handling and anonymization features
- HuggingFace integration components
- Scenario generation and export functionality

### Out of Scope

- Third-party dependencies (report to their respective maintainers)
- System-level vulnerabilities (report to OS vendors)
- Social engineering attacks

## Security Features

TALOS includes several built-in security features:

- **PII Redaction**: Automatic detection and replacement of sensitive information
- **Entropy-based Detection**: Identifies potential API keys and tokens
- **Path Anonymization**: Removes filesystem paths that could expose user environments
- **Secure Defaults**: Strict anonymization enabled by default

## Best Practices

When using TALOS:

- Always use `--anonymize-level strict` for production datasets
- Review anonymized outputs before public distribution
- Use `--skip-redact` only for trusted, non-sensitive data
- Keep dependencies updated: `pip install -r requirements.txt --upgrade`
- Use virtual environments to avoid dependency conflicts
