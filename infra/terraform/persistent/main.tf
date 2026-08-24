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


# Persistent Elastic IP
#
# This resource should survive the destruction and recreation
# of the temporary EC2 instance.

resource "aws_eip" "postgres" {
  domain = "vpc"
}


# Persistent IAM role

resource "aws_iam_role" "ec2_role" {
  name = "ec2_s3_read_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}


# Persistent S3 read policy attachment

resource "aws_iam_role_policy_attachment" "s3_read_write_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}


# Persistent instance profile

resource "aws_iam_instance_profile" "postgres_profile" {
  name = "ec2_s3_read_instance_profile"
  role = aws_iam_role.ec2_role.name
}


output "elastic_ip" {
  value = aws_eip.postgres.public_ip
}


output "elastic_ip_allocation_id" {
  value = aws_eip.postgres.id
}


output "instance_profile_name" {
  value = aws_iam_instance_profile.postgres_profile.name
}
