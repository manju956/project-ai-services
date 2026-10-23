#!/usr/bin/env bash
# run.sh — run the IBM i credit-card POC steps inside the forecast-poc-ibmi container.
#
#   ./run.sh build [BASE_IMAGE]   build forecast-poc-ibmi:latest on top of your forecast image
#   ./run.sh check                mapepire smoke test (connection, schema listing)
#   ./run.sh load [args]          create schema/tables on IBM i + load 30,000 rows
#   ./run.sh pull [args]          pull from IBM i -> data/train.parquet, data/holdout.parquet
#   ./run.sh train                AutoGluon training on data/train.parquet
#   ./run.sh serve                start the FastAPI service (container poc-forecast, :8000)
#   ./run.sh infer [args]         score holdout through the API (needs `serve`)
#   ./run.sh infer-local [args]   score holdout in-process, e.g. --write-back
#   ./run.sh verify               check CCDEMO.CC_PREDS on IBM i (row count, confusion matrix, top risks)
#   ./run.sh sql "SELECT ..."     run read-only SQL on IBM i via mapepire (';' separates statements)
#   ./run.sh all                  load -> pull -> train -> infer-local --write-back
#   ./run.sh stop                 stop the API container
#   ./run.sh shell                interactive shell in the container
#
# Env overrides: CONTAINER_ENGINE (podman|docker), IMAGE, ENV_FILE, API_PORT,
#                BASE_IMAGE, WITH_SSH_TUNNEL=1 (for `build`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORECAST_DIR="$(dirname "$HERE")"                 # services/forecast
MAPEPIRE_DIR="$(cd "$FORECAST_DIR/../mapepire" && pwd)"
ENGINE="${CONTAINER_ENGINE:-podman}"
IMAGE="${IMAGE:-forecast-poc-ibmi:latest}"
ENV_FILE="${ENV_FILE:-$HERE/.env}"
API_PORT="${API_PORT:-8000}"
API_NAME="poc-forecast"

die() { echo "ERROR: $*" >&2; exit 1; }

# Fills ARGS with the options shared by every container run.
common_args() {
    [ -f "$ENV_FILE" ] || die "$ENV_FILE not found. Run: cp ../../mapepire/.env.example $ENV_FILE  and edit it (no quotes around values)."
    ARGS=(--env-file "$ENV_FILE"
          -v "$FORECAST_DIR:/app:z"
          -v "$MAPEPIRE_DIR:/opt/mapepire:ro,z"
          -e "PYTHONUNBUFFERED=1")
    # Connection mode is decided ONLY by IBMI_SSH_KEY_HOST_PATH in .env:
    #   set   -> SSH-tunnel mode (key mounted at /run/secrets/ibmi_key, image needs sshtunnel)
    #   unset -> direct mode (IBMI_SSH_KEY_PATH is forced empty inside the container)
    # The connector itself switches to tunnel mode whenever IBMI_SSH_KEY_PATH is non-empty,
    # so we always override it explicitly (-e after --env-file wins) instead of trusting .env.
    # Parsing tolerates leading spaces, an "export " prefix and CRLF line endings.
    local key stray
    key="$(env_val IBMI_SSH_KEY_HOST_PATH)"
    stray="$(env_val IBMI_SSH_KEY_PATH)"
    if [ -n "$key" ]; then
        [ -f "$key" ] || die "IBMI_SSH_KEY_HOST_PATH=$key does not exist"
        "$ENGINE" run --rm "$IMAGE" python -c "import sshtunnel" >/dev/null 2>&1 \
            || die "SSH-tunnel mode (IBMI_SSH_KEY_HOST_PATH is set) but image $IMAGE has no sshtunnel.
  Tunnel mode: WITH_SSH_TUNNEL=1 $0 build <forecast-image>
  Direct mode: remove/comment IBMI_SSH_KEY_HOST_PATH in $ENV_FILE."
        ARGS+=(-v "$key:/run/secrets/ibmi_key:ro,z" -e "IBMI_SSH_KEY_PATH=/run/secrets/ibmi_key")
        echo "[run.sh] connection mode: SSH tunnel (key $key)" >&2
    else
        [ -z "$stray" ] || echo "[run.sh] WARNING: IBMI_SSH_KEY_PATH=$stray is set in $ENV_FILE but IBMI_SSH_KEY_HOST_PATH is not; ignoring it (direct mode)." >&2
        ARGS+=(-e "IBMI_SSH_KEY_PATH=" -e "IBMI_SSH_KEY_PASSPHRASE=")
        echo "[run.sh] connection mode: direct (wss://\$IBMI_HOST:\${IBMI_PORT:-8076}, no SSH tunnel)" >&2
    fi
}

# env_val NAME -> last value of NAME in $ENV_FILE (tolerates indentation, "export ", CRLF)
env_val() {
    sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*=[[:space:]]*//p" "$ENV_FILE" \
        | tail -1 | tr -d '\r'
}

run_in() {            # run_in <workdir> <cmd...>   (one-shot, removed on exit)
    local wd="$1"; shift
    common_args
    mkdir -p "$HERE/data"
    "$ENGINE" run --rm "${ARGS[@]}" -w "$wd" "$IMAGE" "$@"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
    build)
        base="${1:-${BASE_IMAGE:-localhost/forecast:latest}}"
        "$ENGINE" image inspect "$base" >/dev/null 2>&1 \
            || die "base image '$base' not found. Build it first: $ENGINE build -t forecast:latest -f $FORECAST_DIR/Containerfile $FORECAST_DIR  (or pass your image name: ./run.sh build <image>)"
        "$ENGINE" build -t "$IMAGE" -f "$HERE/Containerfile" \
            --build-arg "BASE_IMAGE=$base" \
            --build-arg "WITH_SSH_TUNNEL=${WITH_SSH_TUNNEL:-0}" "$HERE"
        ;;
    check)       run_in /app/poc_ibmi python /opt/mapepire/example_query.py ;;
    load)        run_in /app/poc_ibmi python load_to_ibmi.py "$@" ;;
    pull)        run_in /app/poc_ibmi python pull_from_ibmi.py "$@" ;;
    train)
        [ -f "$HERE/data/train.parquet" ] || die "data/train.parquet missing - run: ./run.sh pull"
        run_in /app env TRAIN_DATA_PATH=/app/poc_ibmi/data/train.parquet python trainer_ag.py "$@"
        ;;
    serve)
        [ -f "$FORECAST_DIR/ag_forecast_model/predictor.pkl" ] || die "no trained model - run: ./run.sh train"
        common_args
        "$ENGINE" rm -f "$API_NAME" >/dev/null 2>&1 || true
        "$ENGINE" run -d "${ARGS[@]}" --name "$API_NAME" -p "$API_PORT:8000" -w /app "$IMAGE" \
            uvicorn forecast_ag:app --host 0.0.0.0 --port 8000
        echo "API starting on http://localhost:$API_PORT  (logs: $ENGINE logs -f $API_NAME)"
        ;;
    infer)
        # --network host so 127.0.0.1:$API_PORT reaches the published API port
        common_args
        mkdir -p "$HERE/data"
        "$ENGINE" run --rm --network host "${ARGS[@]}" -w /app/poc_ibmi "$IMAGE" \
            python infer_from_ibmi.py --mode api --api-url "http://127.0.0.1:$API_PORT/forecast" "$@"
        ;;
    infer-local) run_in /app/poc_ibmi python infer_from_ibmi.py --mode local "$@" ;;
    verify)      run_in /app/poc_ibmi python run_sql.py --verify ;;
    sql)         [ $# -ge 1 ] || die "usage: ./run.sh sql \"SELECT ...\""
                 run_in /app/poc_ibmi python run_sql.py "$@" ;;
    all)
        "$0" load && "$0" pull && "$0" train && "$0" infer-local --write-back
        ;;
    stop)        "$ENGINE" rm -f "$API_NAME" ;;
    shell)       common_args; "$ENGINE" run --rm -it "${ARGS[@]}" -w /app/poc_ibmi "$IMAGE" bash ;;
    *)           sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
