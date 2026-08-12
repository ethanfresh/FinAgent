# Uses the account's default VPC/subnets to keep this reference deployment
# small. A production rollout should replace this with a dedicated VPC
# (private subnets for nodes, public subnets for the ALB) via a module like
# terraform-aws-modules/vpc/aws.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
