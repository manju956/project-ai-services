# PostgreSQL 18 + pgvector Container Image

This directory contains a custom PostgreSQL 18 container image based on Red Hat Universal Base Image 9 (UBI 9),
with the [pgvector](https://github.com/pgdl/pgvector) extension pre-installed for vector similarity search.

> **Note**: This image is based on the [official PostgreSQL Docker images](https://github.com/docker-library/postgres) maintained by the Docker Community. The entrypoint scripts and initialization logic are adapted from that project.

## Overview

This PostgreSQL image is designed for production use with the following features:

- **Base Image**: Red Hat UBI 9 Minimal
- **PostgreSQL Version**: 18
- **pgvector Version**: latest `pgvector_18` from PGDG RPM repo (≥ 0.7.0)
- **Architecture Support**: ppc64le, x86_64, aarch64
- **User Management**: Runs as non-root `postgres` user (UID/GID 26)
- **Data Directory**: `/var/lib/pgsql/18/data` (compatible with pg_ctlcluster)
- **Volume Mount**: `/var/lib/pgsql`

## Features

### pgvector Extension
- Installed from the official PGDG RPM (`pgvector_18`) — no source compilation required
- `CREATE EXTENSION vector` runs automatically on first container start via `/docker-entrypoint-initdb.d/001-pgvector.sql`
- Provides `vector` data type (up to 16 000 dimensions) and index methods:
  - `ivfflat` — inverted file index, fast approximate search with tunable recall
  - `hnsw` — Hierarchical Navigable Small World, higher recall, larger build memory footprint
- Supports `<->` (L2), `<#>` (inner product), `<=>` (cosine) distance operators

### Security
- Runs as non-root user by default
- Uses `gosu` compiled from source (no prebuilt binary) for privilege de-escalation
- Configurable authentication methods via environment variables

### Initialization
- Automatic database initialization on first run
- Support for initialization scripts (`.sh`, `.sql`, `.sql.gz`, `.sql.xz`, `.sql.zst`)
- Custom initialization directory: `/docker-entrypoint-initdb.d/`

### Configuration
- Pre-configured to listen on all interfaces
- Customizable via environment variables
- Support for custom `postgresql.conf` settings

## Building the Image

### Prerequisites
- Podman or Docker installed
- Access to Red Hat UBI repositories
- Access to PostgreSQL YUM repositories

### Build Command

```bash
make build
```

Or manually:

```bash
podman build -t postgres:18-pgvector .
```

### Build for ppc64le Architecture

The image is built natively on ppc64le — no cross-compilation or QEMU required. The CI
pipeline (`ubuntu-24.04-ppc64le` runner) builds it directly. To build locally on a ppc64le
host:

```bash
make build
# or
podman build -t postgres:18-4 .
```

### Verify pgvector

```bash
make pgvector-test
# or
podman run --rm -e POSTGRES_PASSWORD=test icr.io/ai-services/postgres:18-4 \
  bash -c 'ls /usr/pgsql-18/lib/vector.so && echo OK'
```

## Running the Container

### Basic Usage

```bash
podman run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -v pgdata:/var/lib/pgsql \
  icr.io/ai-services/postgres:18-4
```

### Verify pgvector is working after start

```bash
podman exec -it postgres psql -U postgres -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
# expected output: a version string such as 0.8.0
```

### Store and query vectors

```sql
-- create a table with a 1536-dim embedding column
CREATE TABLE embeddings (
    id     BIGSERIAL PRIMARY KEY,
    source TEXT,
    emb    vector(1536)
);

-- add an HNSW index for cosine similarity search
CREATE INDEX ON embeddings USING hnsw (emb vector_cosine_ops);

-- nearest-neighbour query
SELECT source, emb <=> '[0.1, 0.2, ...]'::vector AS distance
FROM   embeddings
ORDER  BY distance
LIMIT  10;
```

### With Custom Database and User (Vector DB use-case)

```bash
podman run -d \
  --name postgres \
  -e POSTGRES_DB=rag_vectors \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=mypassword \
  -p 5432:5432 \
  -v pgdata:/var/lib/pgsql \
  icr.io/ai-services/postgres:18-4
```

> **Note**: `CREATE EXTENSION IF NOT EXISTS vector` runs automatically against
> `POSTGRES_DB` on first start. No manual step is required.

## Environment Variables

### Required Variables

- `POSTGRES_PASSWORD`: Password for the PostgreSQL superuser (required unless using `trust` authentication)

### Optional Variables

- `POSTGRES_USER`: PostgreSQL superuser name (default: `postgres`)
- `POSTGRES_DB`: Default database name (default: same as `POSTGRES_USER`)
- `POSTGRES_INITDB_ARGS`: Additional arguments to pass to `initdb`
- `POSTGRES_INITDB_WALDIR`: Custom location for transaction log directory
- `POSTGRES_HOST_AUTH_METHOD`: Authentication method for host connections (default: `scram-sha-256` for PG 14+)

### Using Docker Secrets

For sensitive data, you can use file-based environment variables:

```bash
podman run -d \
  --name postgres \
  -e POSTGRES_PASSWORD_FILE=/run/secrets/postgres-passwd \
  -v ./secrets/postgres-passwd:/run/secrets/postgres-passwd:ro \
  postgres:18
```

## Initialization Scripts

Place initialization scripts in `/docker-entrypoint-initdb.d/`:

- `*.sh`: Shell scripts (executed or sourced based on permissions)
- `*.sql`: SQL scripts
- `*.sql.gz`: Gzipped SQL scripts
- `*.sql.xz`: XZ-compressed SQL scripts
- `*.sql.zst`: Zstandard-compressed SQL scripts

Example:

```bash
podman run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -v ./init-scripts:/docker-entrypoint-initdb.d:ro \
  -v pgdata:/var/lib/pgsql \
  postgres:18
```

## Data Persistence

The image uses `/var/lib/pgsql` as the volume mount point. PostgreSQL data is stored in `/var/lib/pgsql/18/data`.

This structure allows for easier upgrades using `pg_upgrade --link` without mount point boundary issues.

## Upgrading from Previous Versions

If upgrading from PostgreSQL versions < 18, you'll need to use `pg_upgrade`. The recommended approach:

1. Mount `/var/lib/pgsql` as a single volume
2. Use `pg_upgrade --link` for efficient upgrades
3. See [PostgreSQL upgrade documentation](https://www.postgresql.org/docs/current/pgupgrade.html)

## Helper Scripts

### docker-entrypoint.sh

Main entrypoint script that handles:
- Database initialization
- User and permission setup
- Initialization script processing
- Server startup

### docker-ensure-initdb.sh

Utility script for Kubernetes init containers or CI/CD:
- Ensures database is initialized
- No-op if already initialized
- Can be used as `docker-enforce-initdb.sh` to error if database exists

## Port

- **5432**: PostgreSQL server port (exposed by default)

## Signals

The container uses `SIGINT` as the stop signal, which triggers PostgreSQL's "Fast Shutdown mode":
- New connections are disallowed
- In-progress transactions are aborted
- PostgreSQL stops cleanly and flushes tables to disk

**Note**: Consider setting `--stop-timeout` to 90+ seconds for graceful shutdown of large databases.

## Health Checks

Example health check:

```bash
podman run -d \
  --name postgres \
  --health-cmd="pg_isready -U postgres" \
  --health-interval=10s \
  --health-timeout=5s \
  --health-retries=5 \
  -e POSTGRES_PASSWORD=mysecretpassword \
  postgres:18
```

## Troubleshooting

### Database Won't Start

1. Check logs: `podman logs postgres`
2. Verify permissions on data directory
3. Ensure `POSTGRES_PASSWORD` is set (unless using `trust` authentication)

### Permission Denied Errors

Ensure the volume mount has correct permissions:

```bash
podman unshare chown -R 26:26 /path/to/pgdata
```

### Old Database Format Detected

If upgrading from pre-18 versions, you'll see an error about old database format. Use `pg_upgrade` to migrate your data.

## Security Considerations

- **Never use `POSTGRES_HOST_AUTH_METHOD=trust` in production**
- Use strong passwords or certificate-based authentication
- Consider using PostgreSQL's SSL/TLS support
- Regularly update to the latest PostgreSQL patch version
- Use network policies to restrict database access

## License

See the [LICENSE](../LICENSE) file in the parent directory.

## Credits

This image is based on the [official PostgreSQL Docker images](https://github.com/docker-library/postgres) maintained by the Docker Community. We gratefully acknowledge their work in creating and maintaining the entrypoint scripts and initialization logic.

## References

- [PostgreSQL Official Documentation](https://www.postgresql.org/docs/18/)
- [pgvector GitHub](https://github.com/pgdl/pgvector) — open-source vector similarity search for PostgreSQL
- [PGDG RPM Repository](https://yum.postgresql.org/) — official PostgreSQL RPM packages including `pgvector_18`
- [PostgreSQL Docker Library](https://github.com/docker-library/postgres) - Original source for entrypoint scripts
- [Red Hat UBI](https://www.redhat.com/en/blog/introducing-red-hat-universal-base-image)