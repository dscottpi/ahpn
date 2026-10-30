# AHPN document scraper

Downloads public documents from the [Digital Archive of the Guatemalan National Police Historical Archive](https://ahpn.lib.utexas.edu/) and stores each document's metadata and page images.

The first target is a single public, one-page document: [document 2127283](https://ahpn.lib.utexas.edu/document/2127283).

## Setup

```bash
uv sync
uv run ahpn scrape
```

Files land in `data/documents/2127283/`.

## What gets saved

- `record.json` — document metadata, archive breadcrumbs, and the list of page images
- `raw/document.json`, `raw/breadcrumbs.json`, `raw/manifest.json` — API responses
- `pages/0001.jpg` — the page image from the archive's IIIF service

The scraper calls the public JSON API at `ahpn-api-prod.gsc1-pub.lib.utexas.edu` and the IIIF image URL the site viewer uses. It sends an identifying user agent, waits one second between requests, and downloads a document only when its `visibility` is `PUB`.

## Chunking

Amazon Textract (`DetectDocumentText`) reads each scraped page image into lines. Those lines are grouped into overlapping chunks of about 1,500 characters and stored in Postgres. The database holds the document, its pages, the OCR lines, and the chunks, so a chunk stays tied to the page it came from and can be rebuilt without another Textract call.

Postgres is the store for the chunks. A search index can sit in front of it later; the lines and chunks do not need a separate service to start.

```bash
docker compose up -d
uv run ahpn db-init
uv run ahpn chunk --document-id 2127283
```

`chunk` creates the tables if they are missing. It uses AWS credentials from the standard chain and reads images from `data/documents`. A second run reuses the stored OCR lines. Pass `--force` to call Textract again.

`AHPN_DATABASE_URL` overrides the default `postgresql+psycopg://ahpn:ahpn@localhost:5433/ahpn`. The Compose file publishes Postgres on 5433 so it does not collide with a Postgres already listening on 5432. `AHPN_CHUNK_CHARS` and `AHPN_CHUNK_OVERLAP` change the chunk size.

## S3

Local files are the default. The same keys can be written to S3 when this runs on AWS:

```bash
uv run ahpn scrape --storage s3 --s3-bucket YOUR_BUCKET
```

Credentials come from the standard AWS chain (environment variables, shared config, or an instance role). `--s3-prefix` defaults to `documents`. Set `AWS_REGION` for the bucket region.

`AHPN_STORAGE`, `AHPN_S3_BUCKET`, `AHPN_S3_PREFIX`, `AHPN_DATA_DIR`, `AHPN_API_BASE_URL`, and `AHPN_REQUEST_DELAY` override the same settings.

## Running on AWS

`infra/stack.yaml` is a CloudFormation stack for a proof of concept that runs the whole flow for one document on AWS:

- an S3 bucket for the scraped records and page images
- RDS Postgres (`db.t4g.micro`, private) for documents, pages, OCR lines, and chunks; RDS keeps the password in Secrets Manager
- an ECR repository for the image built from `Dockerfile`
- an ECS cluster and a Fargate (ARM64) task definition whose role can call Textract and read and write the bucket
- a CloudWatch log group, `/ahpn/pipeline`

The task runs in the account's default VPC with a public IP so it can reach the archive, S3, and Textract without a NAT gateway. The database accepts connections only from the task's security group.

```bash
aws login                       # or set AWS_PROFILE
export AWS_REGION=us-east-2     # default when unset
scripts/aws-deploy.sh           # create/update the stack, build and push the image (~10 min the first time)
scripts/aws-run.sh 2127283      # scrape -> Textract -> chunk -> show, then print the task logs
scripts/aws-teardown.sh         # delete everything, including the database
```

`aws-run.sh` starts one Fargate task that runs `ahpn scrape`, `ahpn chunk`, and `ahpn show` for the document, waits for it to stop, and prints its logs. A second run re-downloads the pages and reuses the stored OCR lines.

In the task, the database connection comes from `AHPN_DB_HOST`, `AHPN_DB_PORT`, `AHPN_DB_NAME`, `AHPN_DB_USER`, and `AHPN_DB_PASSWORD`, which are used when `AHPN_DATABASE_URL` is not set. Set `AHPN_VPC_ID` and `AHPN_SUBNET_IDS` before deploying if the region has no default VPC, and `AHPN_STACK` to change the stack name from `ahpn-poc`.
