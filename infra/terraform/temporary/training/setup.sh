#!/bin/bash
set -Eeuo pipefail

# The Deep Learning Base AMI already carries the NVIDIA driver, so unlike the
# ingestion box there is no Docker/Postgres bootstrap here -- training reads its
# dataset straight out of S3.

REPO_DIR=/opt/classification-pipeline
SETUP_LOG=/var/log/setup.log
LOG_BUCKET=cr-games-bucket
LOG_PREFIX=setup-logs

# -------------------------------------------------------------------
# Step 0: Durable logging
#
# cloud-init's own output only reaches the serial console, which dies with the
# instance -- and this box has wedged hard enough in the past that neither SSM
# nor SSH could be used to read it. Everything below is teed to a file that gets
# shipped to S3 on both success and failure, so a post-mortem survives the box.
# -------------------------------------------------------------------
exec > >(tee -a "$SETUP_LOG") 2>&1

# IMDSv2 -- the instance sets http_tokens = "required", so an unauthenticated
# GET against 169.254.169.254 returns 401.
IMDS_TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)

upload_log() {
    # The persistent instance profile already carries AmazonS3FullAccess, so
    # this needs no new IAM. `|| true` because a failed upload must never mask
    # the real exit status.
    aws s3 cp "$SETUP_LOG" "s3://${LOG_BUCKET}/${LOG_PREFIX}/${INSTANCE_ID}.log" || true
}

on_error() {
    local rc=$?
    echo "==> setup.sh FAILED with exit code ${rc}"
    # `wait_for_setup_script` polls for this sentinel over SSM and fails the
    # flow immediately rather than burning the full 1800s timeout. Falls back to
    # /var/log if the clone never happened.
    mkdir -p "$REPO_DIR" 2>/dev/null || true
    echo "$rc" > "${REPO_DIR}/.setup-failed" 2>/dev/null || echo "$rc" > /var/log/.setup-failed
    upload_log
}
trap on_error ERR

# -------------------------------------------------------------------
# Step 1: Swap
#
# g6.xlarge is 4 vCPU / 16 GiB with no swap by default. Unpacking the CUDA
# wheels below has previously driven the box so far into memory pressure that
# the SSM agent stopped heartbeating and sshd stopped accepting -- an
# unreachable instance that still had to be killed from the console. 8 GiB of
# swap on the 200 GB root is free insurance against that.
# -------------------------------------------------------------------
echo "==> Creating swap..."
if ! swapon --show | grep -q /swapfile; then
    fallocate -l 8G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
fi
free -h

# -------------------------------------------------------------------
# Step 2: Verify the GPU is visible
# -------------------------------------------------------------------
echo "==> Verifying NVIDIA driver..."
nvidia-smi

# -------------------------------------------------------------------
# Step 3: Install git and uv, clone the repo onto this ec2 machine
# -------------------------------------------------------------------
echo "==> Installing git and uv..."

dnf install -y git

curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

echo "==> Installing project..."
mkdir -p /opt
git clone https://github.com/J-Mango-19/win_predictor_pipeline.git \
    "$REPO_DIR"

# -------------------------------------------------------------------
# Step 4: Pre-install dependencies
#
# The ingestion box lets the first `uv run` install implicitly, but training
# pulls multi-GB CUDA wheels. Doing that here rather than inside the SSM command
# keeps it off the training run's clock.
#
# The environment below is what keeps this from wedging the box. Amazon Linux
# 2023 mounts /tmp as tmpfs, so uv's default scratch space is charged straight
# against RAM; pointing UV_CACHE_DIR and TMPDIR at the EBS root moves several GB
# of wheel extraction onto disk. The concurrency caps stop uv from decompressing
# torch, cudnn, cublas and nccl in parallel on a 16 GiB box.
# -------------------------------------------------------------------
echo "==> Installing training dependencies (this pulls several GB of CUDA wheels)..."
mkdir -p /opt/uv-cache /opt/tmp
export UV_CACHE_DIR=/opt/uv-cache
export TMPDIR=/opt/tmp
export UV_CONCURRENT_DOWNLOADS=2
export UV_CONCURRENT_INSTALLS=2

cd "${REPO_DIR}/services/training"
# --frozen: uv.lock is committed, so skip resolution entirely.
uv sync --frozen

echo "==> Verifying installation..."
command -v git
command -v uv
test -d "${REPO_DIR}/services/training"

# The definitive driver/CUDA compatibility check: torch's wheels bundle their
# own CUDA 13 runtime, but it still needs a new enough driver underneath. Fail
# here -- before the sentinel -- rather than hours into a paid GPU run.
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

echo "==> Project setup complete!"
# `set -e` means this sentinel only appears on full success. It is the same path
# `wait_for_setup_script` polls for, so it must not change.
touch "${REPO_DIR}/.setup-complete"
upload_log

# step 5: let prefect run the training command remotely
