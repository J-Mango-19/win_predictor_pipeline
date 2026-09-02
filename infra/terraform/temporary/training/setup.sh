#!/bin/bash
set -e

# The Deep Learning Base AMI already carries the NVIDIA driver, so unlike the
# ingestion box there is no Docker/Postgres bootstrap here -- training reads its
# dataset straight out of S3.

# -------------------------------------------------------------------
# Step 1: Verify the GPU is visible
# -------------------------------------------------------------------
echo "==> Verifying NVIDIA driver..."
nvidia-smi

# -------------------------------------------------------------------
# Step 2: Install git and uv, clone the repo onto this ec2 machine
# -------------------------------------------------------------------
echo "==> Installing git and uv..."

dnf install -y git

curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

echo "==> Installing project..."
mkdir -p /opt
git clone https://github.com/J-Mango-19/win_predictor_pipeline.git \
    /opt/classification-pipeline

# -------------------------------------------------------------------
# Step 3: Pre-install dependencies
#
# The ingestion box lets the first `uv run` install implicitly, but training
# pulls multi-GB CUDA wheels. Doing that here rather than inside the SSM command
# keeps it off the training run's clock.
# -------------------------------------------------------------------
echo "==> Installing training dependencies (this pulls several GB of CUDA wheels)..."
cd /opt/classification-pipeline/services/training
uv sync

echo "==> Verifying installation..."
command -v git
command -v uv
test -d /opt/classification-pipeline/services/training

# The definitive driver/CUDA compatibility check: torch's wheels bundle their
# own CUDA 13 runtime, but it still needs a new enough driver underneath. Fail
# here -- before the sentinel -- rather than hours into a paid GPU run.
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

echo "==> Project setup complete!"
# `set -e` means this sentinel only appears on full success. It is the same path
# `wait_for_setup_script` polls for, so it must not change.
touch /opt/classification-pipeline/.setup-complete

# step 4: let prefect run the training command remotely
