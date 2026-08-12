output "cluster_endpoint" {
  value = aws_eks_cluster.this.endpoint
}

output "cluster_name" {
  value = aws_eks_cluster.this.name
}

output "app_irsa_role_arn" {
  description = "Annotate the finagent service account with this ARN"
  value       = aws_iam_role.finagent_irsa.arn
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}
