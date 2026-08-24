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


variable "elastic_ip_allocation_id" {
  description = "Allocation ID of the persistent Elastic IP"
  type        = string
}

variable "instance_profile_name" {
  description = "Name of the persistent EC2 instance profile"
  type        = string
}


resource "aws_security_group" "postgres" {
  name        = "postgres-temporary"
  description = "Security group for temporary PostgreSQL EC2 instance"

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


data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name = "name"

    values = [
      "al2023-ami-2023.*-x86_64"
    ]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}


resource "aws_instance" "postgres" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = "m7i-flex.large"

  iam_instance_profile = var.instance_profile_name

  key_name = "Test Key Pair"

  vpc_security_group_ids = [
    aws_security_group.postgres.id
  ]

  user_data = file("${path.module}/setup.sh")

  tags = {
    Name = "postgres-temporary"
  }
}


# This resource is temporary.
#
# The Elastic IP itself lives in the persistent Terraform
# configuration. This resource merely attaches it to the
# temporary EC2 instance.

resource "aws_eip_association" "postgres" {
  instance_id   = aws_instance.postgres.id
  allocation_id = var.elastic_ip_allocation_id
}


output "instance_id" {
  value = aws_instance.postgres.id
}

output "elastic_ip_allocation_id" {
  value = var.elastic_ip_allocation_id
}
