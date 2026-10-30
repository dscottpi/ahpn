# Shared settings for the aws-*.sh scripts. Source, don't run.

export AWS_REGION="${AWS_REGION:-us-east-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"
STACK="${AHPN_STACK:-ahpn-poc}"

require_aws_session() {
  if ! aws sts get-caller-identity --query Arn --output text >/dev/null 2>&1; then
    echo "AWS credentials are missing or expired. Run \`aws login\` (or set AWS_PROFILE) and try again." >&2
    exit 1
  fi
}

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
