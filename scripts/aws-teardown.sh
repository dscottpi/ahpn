#!/usr/bin/env bash
# Delete the stack, including the S3 objects, the ECR images, and the RDS database (no final snapshot).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/aws-common.sh

require_aws_session

bucket="$(stack_output BucketName)"
read -r -p "Delete stack $STACK in $AWS_REGION, bucket $bucket, and its database? [y/N] " answer
[[ "$answer" == "y" || "$answer" == "Y" ]] || exit 1

aws s3 rm "s3://$bucket" --recursive
aws cloudformation delete-stack --stack-name "$STACK"
echo "Waiting for stack deletion..."
aws cloudformation wait stack-delete-complete --stack-name "$STACK"
echo "Deleted $STACK."
