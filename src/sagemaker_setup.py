import sagemaker
import boto3
# import os

def setup_sagemaker():
    # os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'
    sess = sagemaker.Session()
    sagemaker_session_bucket = None # used for uploading data, models, logs
    if sagemaker_session_bucket is None and sess is not None:
        sagemaker_session_bucket = sess.default_bucket()
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='llm_finetune')['Role']['Arn']

    sess = sagemaker.Session(default_bucket = sagemaker_session_bucket)

    print(f"sagemaker role arn: {role}")
    print(f"sagemaker bucket: {sess.default_bucket()}")
    print(f"sagemaker session region: {sess.boto_region_name}")

    return sess, role
