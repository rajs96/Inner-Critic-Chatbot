import boto3

def load_data_from_s3(bucket_name: str, obj_key: str, region_name) -> bytes:
    """Load data from S3"""
    s3 = boto3.client('s3', region_name=region_name)
    response = s3.get_object(Bucket=bucket_name, Key=obj_key)
    return response['Body'].read()