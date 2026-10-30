#!/usr/bin/env bash
# Run scrape -> Textract -> chunk -> show for one document as a Fargate task and print its logs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/aws-common.sh

document_id="${1:-2127283}"
if [[ ! "$document_id" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [document_id]" >&2
  exit 2
fi

require_aws_session

cluster="$(stack_output ClusterName)"
task_definition="$(stack_output TaskDefinitionArn)"
security_group="$(stack_output TaskSecurityGroupId)"
subnets="$(stack_output SubnetIds)"
log_group="$(stack_output LogGroupName)"

steps="ahpn scrape --document-id $document_id && ahpn chunk --document-id $document_id && ahpn show --document-id $document_id"
overrides="{\"containerOverrides\":[{\"name\":\"ahpn\",\"command\":[\"sh\",\"-c\",\"$steps\"]}]}"

task_arn="$(aws ecs run-task \
  --cluster "$cluster" \
  --task-definition "$task_definition" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$subnets],securityGroups=[$security_group],assignPublicIp=ENABLED}" \
  --overrides "$overrides" \
  --query 'tasks[0].taskArn' --output text)"
task_id="${task_arn##*/}"
echo "Started task $task_id for document $document_id. Waiting for it to finish..."

aws ecs wait tasks-stopped --cluster "$cluster" --tasks "$task_arn"

read -r exit_code stopped_reason < <(aws ecs describe-tasks --cluster "$cluster" --tasks "$task_arn" \
  --query 'tasks[0].[containers[0].exitCode, stoppedReason]' --output text)

echo "----- logs ($log_group, pipeline/ahpn/$task_id)"
aws logs tail "$log_group" --log-stream-names "pipeline/ahpn/$task_id" --since 1h --format short || true
echo "-----"
echo "exit code: $exit_code (${stopped_reason})"
[[ "$exit_code" == "0" ]]
