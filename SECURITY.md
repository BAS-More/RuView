# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in RuView, please report it responsibly:

1. **Do NOT open a public issue** for security vulnerabilities
2. Email the maintainers at security@ruview.dev (or open a private security advisory on GitHub)
3. Include a description of the vulnerability, steps to reproduce, and potential impact
4. Allow up to 72 hours for an initial response

## Security Measures

- All CSI data is processed locally by default — no cloud transmission
- ESP32 firmware uses NVS encryption for stored credentials
- WebSocket connections support TLS
- API endpoints require JWT authentication in production mode
- See `docs/edge-modules/security.md` for edge deployment security details
