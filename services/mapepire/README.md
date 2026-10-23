# mapepire Python Client Container

Lightweight container image that wraps the official [`mapepire-python`](https://mapepire-ibmi.github.io/guides/usage/python/) client and connects to a **mapepire-server** RPM already running on IBM i.

---

## Prerequisites

| What | Where |
|------|-------|
| `mapepire-server` RPM | Installed and running on IBM i (port **8076** by default) |
| `podman` or `docker` | On the machine where you build / run this container |
| IBM i user profile | With `*USE` authority to `mapepire-server` and the DB objects you want to query |
| SSH public key added to IBM i | `~/.ssh/authorized_keys` for the IBM i user profile (Mode A) |

---

## Authentication modes

### Mode A — SSH private key ✅ recommended

`mapepire-python` uses HTTP Basic Auth over WSS — it has no native SSH-key slot.  
The workaround is a **local SSH tunnel**: the container SSHes into IBM i using your private key, forwards `localhost:18076 → IBM i localhost:8076`, then `mapepire-python` connects through that tunnel.  
**No IBM i password is needed.**

```
Container                        IBM i
────────                         ─────
sshtunnel ──SSH(key)──────────→  sshd :22
mapepire-python → 127.0.0.1:18076 ──(tunnel)──→ localhost:8076 (mapepire-server)
```

### Mode B — password

Direct WSS connection using IBM i username + password. No SSH key needed.

---

## Directory layout

```
services/mapepire/
├── Containerfile          ← UBI 9 minimal + Python 3.11 + mapepire-python + sshtunnel
├── mapepire_connector.py  ← DataConnector wrapper (Modes A & B)
├── example_query.py       ← Connection smoke-test (default CMD)
├── .env.example           ← All supported env vars documented
└── README.md              ← This file
```

---

## 1 — Build the image

```bash
cd services/mapepire

podman build -t mapepire-client:latest -f Containerfile .
```

---

## 2 — Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Required
IBMI_HOST=my.ibmi.host
IBMI_USER=MYUSER

# Mode A (SSH key) — comment out IBMI_PASSWORD
IBMI_SSH_KEY_PATH=/run/secrets/ibmi_key

# Mode B (password) — comment out IBMI_SSH_KEY_PATH
# IBMI_PASSWORD=changeme
```

> **Never commit `.env` to source control.** Add it to `.gitignore`.

---

## 3 — Run the smoke-test

### Mode A — SSH private key (no password)

Mount the private key as a read-only secret inside the container:

```bash
podman run --rm \
  --env-file .env \
  -v ~/.ssh/id_rsa:/run/secrets/ibmi_key:ro,Z \
  mapepire-client:latest
```

If your key is passphrase-protected, add:

```bash
  -e IBMI_SSH_KEY_PASSPHRASE=yourpassphrase \
```

### Mode B — password

```bash
podman run --rm \
  -e IBMI_HOST=my.ibmi.host \
  -e IBMI_USER=MYUSER \
  -e IBMI_PASSWORD=secret \
  mapepire-client:latest
```

### Expected output (Mode A, abridged)

```
[INFO] mapepire.example — === mapepire connectivity smoke-test ===
[INFO] mapepire.example — Target: my.ibmi.host:8076 (user=MYUSER)
[INFO] mapepire.example — [1/4] Testing connection …
[INFO] mapepire.example —       SSH tunnel open: local 127.0.0.1:18076 → my.ibmi.host:8076
[INFO] mapepire.example —       Connection OK [SSH-tunnel]: True
[INFO] mapepire.example — [2/4] Listing tables in schema 'QIWS' …
[INFO] mapepire.example —       Found 1 table(s): ['QCUSTCDT']
...
[INFO] mapepire.example — === Smoke-test complete ===
```

---

## 4 — Run a custom SQL query

```bash
podman run --rm \
  --env-file .env \
  -v ~/.ssh/id_rsa:/run/secrets/ibmi_key:ro,Z \
  -e IBMI_TEST_SQL="SELECT * FROM QIWS.QCUSTCDT FETCH FIRST 5 ROWS ONLY" \
  mapepire-client:latest
```

---

## 5 — Use `MapepireConnector` in your own service

```python
from mapepire_connector import MapepireConnector

# Mode A — SSH key (reads IBMI_HOST, IBMI_USER, IBMI_SSH_KEY_PATH from env)
with MapepireConnector() as conn:
    conn.test_connection()
    tables = conn.list_tables("MYLIB")

# Mode A — explicit constructor
with MapepireConnector(
    host="my.ibmi.host",
    username="MYUSER",
    ssh_key_path="/run/secrets/ibmi_key",   # mounted into the container
) as conn:
    columns = conn.list_columns("MYLIB", "ORDERS")

# Mode B — password
with MapepireConnector(
    host="my.ibmi.host",
    username="MYUSER",
    password="secret",
) as conn:
    conn.test_connection()
```

---

## 6 — Parquet extraction (optional)

`extract()` requires `pyarrow`. Add it to the Containerfile `pip install` block and rebuild:

```dockerfile
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "mapepire-python>=0.2.0" \
        "python-dotenv>=1.0.0" \
        "sshtunnel>=0.4.0" \
        "pyarrow>=14.0"        # ← add this line
```

Mount an output directory so the Parquet file is accessible on the host:

```bash
podman run --rm \
  --env-file .env \
  -v ~/.ssh/id_rsa:/run/secrets/ibmi_key:ro,Z \
  -v "$(pwd)/output:/app/output:Z" \
  mapepire-client:latest \
  python -c "
from mapepire_connector import MapepireConnector
with MapepireConnector() as c:
    r = c.extract('QIWS','QCUSTCDT',['CUSNUM','LSTNAM'],'/app/output/out.parquet')
    print(r)
"
```

---

## Environment variable reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `IBMI_HOST` | ✅ | — | IBM i hostname or IP |
| `IBMI_USER` | ✅ | — | IBM i user profile |
| `IBMI_SSH_KEY_PATH` | Mode A | — | Path to SSH private key **inside the container** |
| `IBMI_SSH_KEY_PASSPHRASE` | | — | Passphrase if the key is encrypted |
| `IBMI_SSH_PORT` | | `22` | IBM i sshd port |
| `IBMI_TUNNEL_LOCAL_PORT` | | `18076` | Local tunnel port inside the container |
| `IBMI_PASSWORD` | Mode B | — | IBM i password |
| `IBMI_PORT` | | `8076` | mapepire-server TCP port |
| `IBMI_IGNORE_TLS` | | `false` | Skip TLS cert verification |
| `IBMI_TEST_SCHEMA` | | `QIWS` | Schema to inspect in smoke-test |
| `IBMI_TEST_TABLE` | | `QCUSTCDT` | Table to describe in smoke-test |
| `IBMI_TEST_SQL` | | — | Custom SQL for smoke-test step 4 |
| `LOG_LEVEL` | | `INFO` | Python logging level |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on port 8076 | mapepire-server not running | `STRTCPSVR SERVER(*MAPEPIRE)` on IBM i |
| `Connection refused` on port 22 | IBM i sshd not running | `STRTCPSVR SERVER(*SSHD)` on IBM i |
| `Authentication failed` (SSH) | Key not in IBM i `authorized_keys` | Add your public key to `~MYUSER/.ssh/authorized_keys` on IBM i |
| `No authentication method configured` | Neither `IBMI_SSH_KEY_PATH` nor `IBMI_PASSWORD` set | Set one of these env vars |
| `SSL certificate verify failed` | Self-signed cert | Set `IBMI_IGNORE_TLS=true` |
| `CPF2204` / `not authorized` | Missing DB authority | Grant `*USE` to the IBM i user profile on the library |
| `mapepire-python is not installed` | Image not rebuilt | Re-run `podman build` |
| `sshtunnel is not installed` | Image not rebuilt after Containerfile edit | Re-run `podman build` |
