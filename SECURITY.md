# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions of Thunders AI:

| Version | Supported          | Status       |
| ------- | ------------------ | ------------ |
| 1.0.x   | :white_check_mark: | Current      |
| 0.9.x   | :white_check_mark: | Maintenance  |
| < 0.9   | :x:                | End of life  |

We recommend always using the latest stable release to ensure you have the most up-to-date security patches.

## Reporting a Vulnerability

We take security vulnerabilities seriously and appreciate responsible disclosure. If you discover a security vulnerability in Thunders AI, please follow these steps:

### Reporting Process

1. **Do not** report security vulnerabilities through public GitHub issues.
2. Email security findings to **security@thunders-ai.dev** with the subject line `[SECURITY] Brief Description`.
3. Include the following in your report:
   - Type of vulnerability (e.g., buffer overflow, SQL injection, authentication bypass)
   - Full path of the affected source file(s)
   - Steps to reproduce the vulnerability
   - Proof-of-concept or exploit code (if available)
   - Potential impact of the vulnerability
   - Suggested fix (if you have one)
4. You should receive an acknowledgment within **24 hours**.
5. We will keep you informed of the progress toward a fix and full advisory.

### Response Timeline

- **Initial Response**: Within 24 hours
- **Triage & Confirmation**: Within 72 hours
- **Fix Development**: Depends on severity (critical: 7 days, high: 14 days, medium: 30 days, low: next release)
- **Advisory Publication**: After the fix is released

### Responsible Disclosure

We ask that you:

- Give us a reasonable amount of time to fix the issue before any public disclosure
- Avoid exploiting the vulnerability beyond what is necessary to demonstrate it
- Do not access, modify, or delete other users' data
- Act in good faith to protect user privacy and system integrity

We are committed to acknowledging contributors who responsibly report security issues.

## Security Features

Thunders AI includes several built-in security features designed to protect your data and infrastructure:

### Encryption
- **AES-256-GCM** end-to-end encryption for data at rest and in transit
- Secure key management with hardware security module (HSM) support
- TLS 1.3 for all network communications

### Authentication & Authorization
- **JWT-based** authentication with configurable token expiration
- **Role-based access control (RBAC)** with granular permissions
- API key management with automatic rotation
- OAuth 2.0 / OpenID Connect integration support

### Sandbox Execution
- Isolated execution environment for untrusted code and model inference
- Resource limits (CPU, memory, network) on sandboxed processes
- Filesystem isolation with read-only mounts

### Threat Detection
- Real-time anomaly monitoring for API request patterns
- Rate limiting and DDoS protection
- Input validation and sanitization against injection attacks
- Audit logging of all security-relevant events

## Security Best Practices

When deploying Thunders AI in production, we recommend the following security practices:

### Configuration
- **Never use default secrets** in production — always change all passwords, API keys, and JWT secrets
- Use **environment variables** or a secrets manager (e.g., HashiCorp Vault, AWS Secrets Manager) for sensitive configuration
- Enable **HTTPS/TLS** for all external communications
- Set `THUNDERS_AI_LOG_LEVEL=WARNING` or higher in production to avoid logging sensitive data

### Network
- Deploy behind a **reverse proxy** (Nginx, Cloudflare) with WAF capabilities
- Use **network policies** to restrict pod-to-pod communication in Kubernetes
- Enable **rate limiting** on all public-facing endpoints
- Implement **IP allowlisting** for administrative access

### Data
- Encrypt all **data at rest** using AES-256
- Implement **data retention policies** and automatic purging
- Use **database encryption** (e.g., PostgreSQL with pgcrypto)
- Regularly **back up** and test restoration of critical data

### Monitoring
- Enable **audit logging** for all API calls and administrative actions
- Set up **alerting** for anomalous access patterns
- Conduct regular **security scans** of dependencies (`pip audit`, `safety`)
- Perform periodic **penetration testing**

### Updates
- Keep Thunders AI and all dependencies **up to date**
- Subscribe to our **security advisory** notifications on GitHub
- Review **CHANGELOG.md** for security-related fixes in each release
- Test updates in a **staging environment** before deploying to production

## Dependency Security

We regularly audit our dependencies for known vulnerabilities using:

- [pip-audit](https://pypi.org/project/pip-audit/) for Python package vulnerabilities
- [Safety](https://pypi.org/project/safety/) for dependency security checks
- GitHub Dependabot for automated dependency updates
- Pre-commit hooks that check for vulnerable dependencies

If you discover a vulnerability in one of our dependencies, please report it through the same process outlined above.
