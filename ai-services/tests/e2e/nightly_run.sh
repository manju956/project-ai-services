#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Required CI environment variables (must be set in the Jenkins job):
#
#   CATALOG_PASSWORD      — admin password for the catalog API server
#   CATALOG_SERVER_URL    — optional; auto-discovered if not set
#   REGISTRY_URL          — container registry URL for podman login
#   REGISTRY_USER_NAME    — container registry username
#   REGISTRY_PASSWORD     — container registry password
#   RH_REGISTRY_URL       — Red Hat registry URL (optional)
#   RH_REGISTRY_USER_NAME — Red Hat registry username (optional)
#   RH_REGISTRY_PASSWORD  — Red Hat registry password (optional)
# ---------------------------------------------------------------------------

# Validate that critical variables are present before doing any work.
: "${CATALOG_PASSWORD:?CATALOG_PASSWORD must be set in the Jenkins job}"
: "${REGISTRY_URL:?REGISTRY_URL must be set in the Jenkins job}"
: "${REGISTRY_USER_NAME:?REGISTRY_USER_NAME must be set in the Jenkins job}"
: "${REGISTRY_PASSWORD:?REGISTRY_PASSWORD must be set in the Jenkins job}"

# Perform clean-up
echo "Cleaning up existing repository folder"
rm -rf /root/nightly-run/project-ai-services

# Clone the repository
cd /root/nightly-run
echo "Cloning ai services repository"
git clone https://github.com/IBM/project-ai-services.git
echo "Repository clone successfully"

# Trigger the suite
cd project-ai-services/ai-services
go install github.com/onsi/ginkgo/v2/ginkgo@latest
export PATH=$PATH:$(go env GOPATH)/bin

echo "Triggering the E2E suite run"
# --timeout=0 disables the Go test suite-level timeout entirely.
# Individual spec and node timeouts (SpecTimeout/NodeTimeout decorators in
# e2e_suite_test.go) govern each step:
#   BeforeAll  — NodeTimeout(3h)  covers model download + ingestion + judge start
#   It (eval)  — SpecTimeout(3h)  covers 50-question evaluation loop
# This prevents the suite from being killed before the evaluation completes
# regardless of how slow the LLM is on this hardware.
#
# The env vars below are explicitly forwarded so they survive the make
# subprocess boundary on Jenkins (the outer shell env is not always inherited
# in all Jenkins configurations).
TEST_OUTPUT=$(
  CATALOG_PASSWORD="${CATALOG_PASSWORD}" \
  CATALOG_SERVER_URL="${CATALOG_SERVER_URL:-}" \
  REGISTRY_URL="${REGISTRY_URL}" \
  REGISTRY_USER_NAME="${REGISTRY_USER_NAME}" \
  REGISTRY_PASSWORD="${REGISTRY_PASSWORD}" \
  RH_REGISTRY_URL="${RH_REGISTRY_URL:-}" \
  RH_REGISTRY_USER_NAME="${RH_REGISTRY_USER_NAME:-}" \
  RH_REGISTRY_PASSWORD="${RH_REGISTRY_PASSWORD:-}" \
  make test-generate-report TEST_ARGS="--timeout=0" DELETE_APP=true
)

# Capture the output of the suite
echo "Output of E2E test run"
echo "$TEST_OUTPUT"
