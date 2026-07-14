-- Enable the pgvector extension in the default database.
-- This script runs automatically on first container start via
-- /docker-entrypoint-initdb.d and is a no-op on subsequent starts.
--
-- The extension provides the `vector` data type and index methods
-- (ivfflat, hnsw) needed for ANN similarity search.

CREATE EXTENSION IF NOT EXISTS vector;
