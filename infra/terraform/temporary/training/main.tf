terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

# Unlike the ingestion stack, this one takes no Elastic IP. Training only makes
# outbound calls (S3, Secrets Manager, wandb, PyPI), so it has no need for the
# stable whitelisted address the Clash Royale API requires.
variable "instance_profile_name" {
  description = "Name of the persistent EC2 instance profile"
  type        = string
}

# -----------------------------------------------------------------------------
# Data Sources (New and Existing)
# -----------------------------------------------------------------------------

# Get current AWS account ID and region to dynamically build the Secret ARN
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Look up the existing IAM Instance Profile to get its underlying Role Name
data "aws_iam_instance_profile" "persistent" {
  name = var.instance_profile_name
}

# The Deep Learning Base AMI ships the NVIDIA driver preinstalled, which torch's
# Linux wheels require (they bundle their own CUDA 13 runtime but not a driver).
# Building on plain al2023 would mean a slow, brittle driver install at boot.
data "aws_ssm_parameter" "deep_learning_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
}

# -----------------------------------------------------------------------------
# IAM Policy & Attachment (New)
# -----------------------------------------------------------------------------

resource "aws_iam_policy" "prefect_secrets" {
  # IAM policy names are account-unique, so this cannot reuse the ingestion
  # stack's "prefect_secrets_read_temporary" without colliding whenever both
  # stacks happen to exist at once.
  name        = "prefect_secrets_read_training"
  description = "Allows the persistent EC2 role to read Prefect secrets"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action   = "secretsmanager:GetSecretValue",
      Effect   = "Allow",
      # Scoped strictly to the specific secret in your current region/account
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "prefect_secrets_attach" {
  # Attach the new policy to the Role found inside your persistent Instance Profile
  role       = data.aws_iam_instance_profile.persistent.role_name
  policy_arn = aws_iam_policy.prefect_secrets.arn
}

# -----------------------------------------------------------------------------
# EC2 & Networking Resources
# -----------------------------------------------------------------------------

resource "aws_security_group" "training" {
  name        = "training-temporary"
  description = "Security group for temporary GPU training EC2 instance"

  ingress {
    description = "SSH from my home IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"

    cidr_blocks = [
      "108.44.43.146/32", # home
      "129.74.0.0/16",    # ND primary
      "66.254.224.0/19",  # ND secondary
      "66.205.160.0/20",  # ND secondary
      "129.74.86.0/23",   # ND secondary
    ]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "training" {
  ami                  = data.aws_ssm_parameter.deep_learning_ami.value
  instance_type        = "g6.xlarge" # cheapest L4 offering
  iam_instance_profile = var.instance_profile_name

  key_name = "Test Key Pair"

  vpc_security_group_ids = [
    aws_security_group.training.id
  ]

  # No Elastic IP is associated, so ask for an ephemeral public address to reach
  # S3, PyPI and wandb.
  associate_public_ip_address = true

  # The Deep Learning AMI's own snapshot is 75 GB, so this cannot go below that.
  # The headroom above it holds torch's CUDA 13 wheels (~1 GB of cudnn/nccl/
  # cusparselt/triton alone), the downloaded parquet dataset, and the
  # train/val split copies `make_splits` writes back out to disk.
  root_block_device {
    volume_size           = 200
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = file("${path.module}/setup.sh")

  tags = {
    Name = "training-temporary"
  }
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "instance_id" {
  value = aws_instance.training.id
}
