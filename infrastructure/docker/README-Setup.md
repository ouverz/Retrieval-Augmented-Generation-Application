# Database Setup Guide

## Automatic pgvector Installation

This setup ensures pgvector extension is automatically installed when you replicate the application.

### How it Works

1. **Docker Init Script**: `init-db.sql` runs automatically on first database startup
2. **Extension Installation**: Creates `vector` and `timescaledb` extensions
3. **Health Checks**: Ensures database is ready before application starts

### Setup Steps

```bash
# 1. Copy environment variables
cp .env.example .env

# 2. Add your OpenAI API key to .env
# OPENAI_API_KEY=your_actual_key_here

# 3. Start services
cd src/docker
docker-compose up -d

# 4. Verify extensions are installed
docker exec timescaledb psql -U postgres -d postgres -c "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'timescaledb');"
```

### Verification Commands

```bash
# Check database health
docker-compose ps

# Test pgvector extension
docker exec timescaledb psql -U postgres -d postgres -c "SELECT version();"
docker exec timescaledb psql -U postgres -d postgres -c "\\dx vector"

# Check Redis connection
docker exec rag_redis redis-cli ping
```

### Troubleshooting

If pgvector is missing:
```bash
# Manual installation (fallback)
docker exec timescaledb psql -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Clean Start

To start completely fresh:
```bash
docker-compose down -v  # Removes volumes
docker-compose up -d    # Rebuilds with init script
```

## Production Considerations

- Change `POSTGRES_PASSWORD` in production
- Use secrets management for sensitive values
- Consider using managed TimescaleDB for production
- Monitor extension versions for updates