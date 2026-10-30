#!/usr/bin/env bash
# Create or update the AWS stack, then build and push the pipeline image.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/aws-common.sh

require_aws_session

vpc_id="${AHPN_VPC_ID:-$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)}"
if [[ -z "$vpc_id" || "$vpc_id" == "None" ]]; then
  echo "No default VPC in $AWS_REGION. Set AHPN_VPC_ID and AHPN_SUBNET_IDS (comma-separated public subnets)." >&2
  exit 1
fi
subnet_ids="${AHPN_SUBNET_IDS:-$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$vpc_id" Name=default-for-az,Values=true \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')}"

echo "Deploying stack $STACK to $AWS_REGION (vpc $vpc_id). The first deploy waits on RDS and takes about 10 minutes."
aws cloudformation deploy \
  --stack-name "$STACK" \
  --template-file infra/stack.yaml \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides "VpcId=$vpc_id" "SubnetIds=$subnet_ids"

repo="$(stack_output RepositoryUri)"
echo "Building and pushing $repo:latest"
aws ecr get-login-password | docker login --username AWS --password-stdin "${repo%%/*}"
docker build --platform linux/arm64 -t "$repo:latest" .
docker push "$repo:latest"

echo
echo "Stack outputs:"
aws cloudformation describe-stacks --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
echo "Next: scripts/aws-run.sh 2127283"
