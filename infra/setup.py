"""
infra/setup.py
--------------
Creates all AWS resources for stock-predictor.
Run per phase:
  python setup.py --phase 2   # S3 + CORS
  python setup.py --phase 3   # + SageMaker + Lambda + Secrets Manager
  python setup.py --phase 4   # + EventBridge + DLQ
"""

import argparse
import boto3
from botocore.exceptions import ClientError
import json
import zipfile
import tarfile
import io
import os

REGION = "us-east-1"
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

# Resource names
BUCKET        = f"stock-predictor-{ACCOUNT_ID}"
LAMBDA_NAME   = "stock-predictor-daily-inference"
SM_ENDPOINT   = "stock-predictor-chronos-endpoint"
SM_MODEL      = "stock-predictor-chronos-model"
SM_CONFIG     = "stock-predictor-chronos-config"
EB_RULE       = "stock-predictor-daily-trigger"
DLQ_NAME      = "stock-predictor-dlq"
SECRET_NAME   = "stock-predictor/alpha-vantage-key"
LAMBDA_ROLE   = "stock-predictor-lambda-role"
GITHUB_ORIGIN = "https://SusanLiu0405.github.io"


# ── Phase 2: S3 ───────────────────────────────────────────────────────────────

def create_s3(s3):
    print("Creating S3 bucket...")
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET)
        else:
            s3.create_bucket(
                Bucket=BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        print(f"  Created bucket: {BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  Bucket already exists: {BUCKET}")

    s3.put_bucket_cors(
        Bucket=BUCKET,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedOrigins": [GITHUB_ORIGIN, "http://localhost:8080", "http://127.0.0.1:8080"],
                    "AllowedMethods": ["GET"],
                    "AllowedHeaders": ["*"],
                    "MaxAgeSeconds": 3600,
                }
            ]
        }
    )
    print("  CORS configured.")

    # Block public access OFF so frontend can read JSON
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        }
    )

    s3.put_bucket_policy(
        Bucket=BUCKET,
        Policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicReadPredictions",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{BUCKET}/predictions/*"
                }
            ]
        })
    )
    print("  Bucket policy set (public GET on /predictions/*).")


# ── Phase 3: Secrets Manager + IAM + SageMaker + Lambda ──────────────────────
import boto3
from botocore.exceptions import ClientError
# The secret has been configured in aws secret manager, use this code segment to get it.
def get_secret():

    secret_name = "stock-predictor/alpha-vantage-key"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']
    return secret

def create_lambda_role(iam):
    print("Creating Lambda IAM role...")
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": [
                    "lambda.amazonaws.com",
                    "sagemaker.amazonaws.com",   # needed for SageMaker model execution
                ]
            },
            "Action": "sts:AssumeRole"
        }]
    }
    try:
        role = iam.create_role(
            RoleName=LAMBDA_ROLE,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Role for stock-predictor Lambda"
        )
        role_arn = role["Role"]["Arn"]
        print(f"  Created role: {LAMBDA_ROLE}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=LAMBDA_ROLE)["Role"]["Arn"]
        print(f"  Role already exists: {LAMBDA_ROLE}")

    # Attach managed policies
    for policy in [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/SecretsManagerReadWrite",
    ]:
        iam.attach_role_policy(RoleName=LAMBDA_ROLE, PolicyArn=policy)
    print("  Policies attached.")
    return role_arn


def upload_model_artifact(s3):
    """
    Package a custom inference.py + requirements.txt into model.tar.gz and upload to S3.
    Chronos is a time-series model with no standard HuggingFace task type, so we need
    a custom handler that loads it via the chronos-forecasting package.
    """
    inference_py = '''
import json
import torch
import numpy as np

def model_fn(model_dir):
    # Download Chronos-Bolt from HuggingFace Hub at cold-start
    from chronos import ChronosBoltPipeline
    pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return pipeline

def input_fn(request_body, content_type):
    return json.loads(request_body)

def predict_fn(data, pipeline):
    prices   = data["inputs"]        # list of lists (one per ticker)
    pred_len = data.get("parameters", {}).get("prediction_length", 1)
    context  = [torch.tensor(p, dtype=torch.float32) for p in prices]
    # context and prediction_length are positional args in ChronosBoltPipeline.predict()
    forecast = pipeline.predict(context, pred_len)
    # forecast: list of tensors [n_samples, pred_len] per series — return medians
    medians  = [float(np.median(f.numpy())) for f in forecast]
    return {"predictions": medians}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
'''.strip()

    requirements_txt = "chronos-forecasting\n"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in [("code/inference.py", inference_py),
                               ("code/requirements.txt", requirements_txt)]:
            encoded = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    buf.seek(0)

    key = "sagemaker/chronos/model.tar.gz"
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read())
    uri = f"s3://{BUCKET}/{key}"
    print(f"  Model artifact uploaded: {uri}")
    return uri


def create_sagemaker_endpoint(sm_client, s3_client):
    print("Creating SageMaker Serverless endpoint for Chronos-2...")

    model_uri = upload_model_artifact(s3_client)

    # PyTorch 2.6.0 container — already satisfies chronos-forecasting's torch>=2.2
    # requirement, so pip only downloads the small chronos packages, not a new torch.
    image = (
        f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
        "pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker-v1.67"
    )

    # Model
    try:
        sm_client.create_model(
            ModelName=SM_MODEL,
            PrimaryContainer={
                "Image": image,
                "ModelDataUrl": model_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                }
            },
            ExecutionRoleArn=f"arn:aws:iam::{ACCOUNT_ID}:role/{LAMBDA_ROLE}",
        )
        print(f"  Created model: {SM_MODEL}")
    except (sm_client.exceptions.ResourceInUse, ClientError):
        print(f"  Model already exists: {SM_MODEL}")

    # Endpoint config (Serverless)
    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=SM_CONFIG,
            ProductionVariants=[{
                "VariantName": "AllTraffic",
                "ModelName": SM_MODEL,
                "ServerlessConfig": {
                    "MemorySizeInMB": 3072,
                    "MaxConcurrency": 5,
                }
            }]
        )
        print(f"  Created endpoint config: {SM_CONFIG}")
    except (sm_client.exceptions.ResourceInUse, ClientError):
        print(f"  Endpoint config already exists: {SM_CONFIG}")

    # Endpoint
    try:
        sm_client.create_endpoint(
            EndpointName=SM_ENDPOINT,
            EndpointConfigName=SM_CONFIG,
        )
        print(f"  Creating endpoint: {SM_ENDPOINT} (takes ~5 min, check AWS Console)")
    except (sm_client.exceptions.ResourceInUse, ClientError):
        print(f"  Endpoint already exists: {SM_ENDPOINT}")


def create_lambda(lm, role_arn):
    print("Creating Lambda function...")

    # Package a stub handler so Lambda can be created before real code is ready
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        stub = "def handler(event, context):\n    print('stub — replace with real lambda_handler.py')\n"
        zf.writestr("lambda_handler.py", stub)
    buf.seek(0)

    try:
        lm.create_function(
            FunctionName=LAMBDA_NAME,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_handler.handler",
            Code={"ZipFile": buf.read()},
            Timeout=900,          # 15 min max — Chronos cold start can be slow
            MemorySize=512,
            Environment={
                "Variables": {
                    "S3_BUCKET": BUCKET,
                    "SM_ENDPOINT": SM_ENDPOINT,
                    "REGION": REGION,
                }
            }
        )
        # Set reserved concurrency separately — prevents duplicate runs
        lm.put_function_concurrency(
            FunctionName=LAMBDA_NAME,
            ReservedConcurrentExecutions=1,
        )
        print(f"  Created Lambda: {LAMBDA_NAME}")
        print("  !! Deploy real inference/lambda_handler.py before triggering !!")
    except lm.exceptions.ResourceConflictException:
        print(f"  Lambda already exists: {LAMBDA_NAME}")


# ── Phase 4: EventBridge + DLQ ────────────────────────────────────────────────

def create_eventbridge_with_dlq(eb, sqs, lm):
    print("Creating SQS DLQ...")
    try:
        q = sqs.create_queue(QueueName=DLQ_NAME)
        dlq_url = q["QueueUrl"]
        dlq_arn = sqs.get_queue_attributes(
            QueueUrl=dlq_url, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        print(f"  Created DLQ: {DLQ_NAME}")
    except sqs.exceptions.QueueNameExists:
        dlq_url = sqs.get_queue_url(QueueName=DLQ_NAME)["QueueUrl"]
        dlq_arn = sqs.get_queue_attributes(
            QueueUrl=dlq_url, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        print(f"  DLQ already exists: {DLQ_NAME}")

    print("Creating EventBridge rule (6:30 PM ET = 23:30 UTC)...")
    rule = eb.put_rule(
        Name=EB_RULE,
        ScheduleExpression="cron(30 23 ? * MON-FRI *)",  # weekdays only
        State="ENABLED",
        Description="Triggers stock-predictor Lambda daily after market close"
    )
    rule_arn = rule["RuleArn"]

    lambda_arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_NAME}"

    eb.put_targets(
        Rule=EB_RULE,
        Targets=[{
            "Id": "StockPredictorLambda",
            "Arn": lambda_arn,
            "DeadLetterConfig": {"Arn": dlq_arn},
        }]
    )
    print(f"  EventBridge rule created: {EB_RULE}")

    # Allow EventBridge to invoke Lambda
    try:
        lm.add_permission(
            FunctionName=LAMBDA_NAME,
            StatementId="AllowEventBridge",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
        print("  Lambda permission granted to EventBridge.")
    except lm.exceptions.ResourceConflictException:
        print("  Lambda permission already exists.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[2, 3, 4],
                        help="2=S3, 3=+SageMaker+Lambda+Secrets, 4=+EventBridge+DLQ")
    args = parser.parse_args()

    s3      = boto3.client("s3",         region_name=REGION)
    iam     = boto3.client("iam",        region_name=REGION)
    sm_sec  = boto3.client("secretsmanager", region_name=REGION)
    sm      = boto3.client("sagemaker",  region_name=REGION)
    lm      = boto3.client("lambda",     region_name=REGION)
    eb      = boto3.client("events",     region_name=REGION)
    sqs     = boto3.client("sqs",        region_name=REGION)

    if args.phase >= 2:
        create_s3(s3)

    if args.phase >= 3:
        print("Fetching Alpha Vantage key from Secrets Manager...")
        get_secret()
        print("  Secret verified.")
        role_arn = create_lambda_role(iam)
        create_sagemaker_endpoint(sm, s3)
        create_lambda(lm, role_arn)

    if args.phase >= 4:
        create_eventbridge_with_dlq(eb, sqs, lm)

    print("\nDone.")


if __name__ == "__main__":
    main()