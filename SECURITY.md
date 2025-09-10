# Security Features

This document describes the security measures implemented in the RAG application.

## API Authentication

### API Key Authentication
All protected endpoints require an API key in the request header:

```bash
curl -H "X-API-Key: your_api_key_here" http://localhost:8000/query
```

### Configuration
API keys are configured via the `RAG_API_KEYS` environment variable:

```bash
# Single key
RAG_API_KEYS=abc123def456

# Multiple keys (comma-separated)
RAG_API_KEYS=key1,key2,key3
```

### Generate Secure API Keys
```bash
# Generate a secure 32-byte hex key
openssl rand -hex 32
```

## Rate Limiting

Rate limits are enforced per IP address:

| Endpoint | Rate Limit |
|----------|------------|
| `POST /query` | 10 requests/minute |
| `POST /ingest` | 5 requests/hour |
| `POST /init` | 2 requests/hour |
| `GET /cache/stats` | 30 requests/minute |
| `POST /cache/clear` | 10 requests/hour |
| `GET /cache/health` | 60 requests/minute |

## File Upload Security

### Restrictions
- **File types**: Only PDF files allowed
- **File size**: Maximum 100MB per file
- **Filename validation**: Path traversal prevention
- **Content-Type validation**: MIME type checking

### Security Features
- Filename sanitization
- Duplicate handling with unique naming
- Chunked upload with size verification
- Secure file storage in designated directory

## Security Headers

The following security headers are automatically added to all responses:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'`

## CORS Configuration

CORS is configured to allow requests from:
- `http://localhost:8501` (Streamlit)
- `http://localhost:3000` (React development)
- `http://127.0.0.1:8501`
- `http://127.0.0.1:3000`

## Error Handling

### Security Features
- Generic error messages to prevent information disclosure
- Detailed errors logged server-side with correlation IDs
- No stack traces or internal details exposed to clients

### Error Response Format
```json
{
    "detail": "Internal server error occurred",
    "error_id": "abc12345",
    "message": "Please contact support with this error ID if the problem persists"
}
```

## Environment Variables Security

### Required Variables
- `OPENAI_API_KEY`: OpenAI API key
- `TIMESCALE_SERVICE_URL`: Database connection string

### Optional Security Variables
- `RAG_API_KEYS`: API keys for authentication
- `POSTGRES_PASSWORD`: Database password
- `REDIS_PASSWORD`: Redis password

### Security Best Practices
1. Never commit `.env` files to version control
2. Use different `.env` files for different environments
3. Rotate API keys regularly
4. Use strong, unique passwords
5. Consider using secret management services in production

## Production Security Checklist

### Before Deployment
- [ ] Set strong, unique API keys
- [ ] Configure HTTPS/TLS
- [ ] Set up proper database user permissions
- [ ] Enable database SSL connections
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerting
- [ ] Review and test all endpoints
- [ ] Implement backup and recovery procedures

### Monitoring
- Monitor rate limit violations
- Track authentication failures
- Monitor file upload attempts
- Set up alerts for security events

## API Documentation

### Authentication
Add the API key header to all requests:
```
X-API-Key: your_api_key_here
```

### Public Endpoints (No Authentication Required)
- `GET /healthz` - Basic health check
- `GET /init/status` - Initialization status

### Protected Endpoints (Authentication Required)
- All other endpoints require API key authentication

## Security Contact

For security issues, please follow responsible disclosure practices and contact the development team through appropriate channels.