-- Database initialization script for TimescaleDB with pgvector
-- This script runs automatically when the database container starts for the first time

-- Create the vector extension (required for pgvector operations)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the timescaledb extension (if not already available)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Verify extensions are installed
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'timescaledb');

-- Log successful initialization
SELECT 'Database initialized successfully with pgvector and timescaledb extensions' as status;