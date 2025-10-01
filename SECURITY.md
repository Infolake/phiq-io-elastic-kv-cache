<div align="center">

![ΦQ Logo](docs/assets/logo-phi-q-icon-100.png)

# ΦQ™ PHIQ.IO Elastic KV Cache — Security Policy

**Author:** Dr. Guilherme de Camargo | **Organization:** PHIQ.IO Quantum Technologies (ΦQ™)
**Contact:** security@phiq.io | https://phiq.io

Responsible Disclosure • Security Guidelines

**Camargo Constant:** Δ = φ + π = 4.759627

</div>

---

# Security Policy

## Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** create a public issue
2. Email security concerns to: security@phiq.io
3. Include a detailed description of the vulnerability
4. Allow time for investigation and resolution

## Security Guidelines

### For Contributors

- **Never commit secrets, tokens, or credentials** to the repository
- Use environment variables for sensitive configuration
- Follow secure coding practices in CUDA kernels
- Validate all input parameters in CLI tools

### For Users

- **HuggingFace Authentication**: Configure your own tokens via environment variables
- **GPU Memory**: Be cautious with large sequence lengths to avoid OOM
- **Model Downloads**: Only use trusted model sources

### Environment Security

- This project does not include any hardcoded secrets
- Users must configure their own HuggingFace tokens for private models
- All benchmark data is synthetic or uses public models

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Response Timeline

- **Critical vulnerabilities**: 24-48 hours
- **High severity**: 1 week
- **Medium/Low severity**: 2-4 weeks

We will acknowledge receipt of vulnerability reports within 24 hours and provide regular updates on the investigation progress.
