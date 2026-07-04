# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| 1.0.x | Yes |
| < 1.0 | No |

## Reporting a Vulnerability

If you discover a security issue in Meta Analyst, please report it privately:

1. Do **not** open a public GitHub issue for security vulnerabilities.
2. Contact the repository maintainer directly with:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. Allow reasonable time for a fix before public disclosure.

## Security Practices

- Store `OPENAI_API_KEY` and `APP_PASSWORD` only in `.env` or secure environment variables.
- Never commit secrets, API keys, or client ad data to version control.
- Rotate credentials if they may have been exposed.
- Use strong, unique values for `APP_PASSWORD` in production deployments.

## Data Handling

- Uploaded Meta Ads reports are processed in-memory during the Streamlit session.
- No persistent storage of uploaded files is configured by default.
- Review your deployment environment for additional logging or caching behavior.
