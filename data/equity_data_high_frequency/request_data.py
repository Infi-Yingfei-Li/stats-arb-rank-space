#%%
import boto3, os, tqdm
from botocore.config import Config
import datetime
import matplotlib.pyplot as plt

IS_ALL_DATA = False
DATA_TYPE = "minute_aggs_v1"
#DATA_TYPE = "day_aggs_v1"

# Initialize a session using your credentials
session = boto3.Session(
   aws_access_key_id='991fece7-ce74-4421-a8de-a46a134d1ca5',
   aws_secret_access_key='Vpri9DW313MwLIKIRpiCzL5WeAVLs3Xi',
)

#%%
# Create a client with your session and specify the endpoint
s3 = session.client(
   's3',
   endpoint_url='https://files.polygon.io',
   config=Config(signature_version='s3v4'),
)

# List Example
# Initialize a paginator for listing objects
paginator = s3.get_paginator('list_objects_v2')

# Choose the appropriate prefix depending on the data you need:
# - 'global_crypto' for global cryptocurrency data
# - 'global_forex' for global forex data
# - 'us_indices' for US indices data
# - 'us_options_opra' for US options (OPRA) data
# - 'us_stocks_sip' for US stocks (SIP) data
prefix = 'us_stocks_sip'  # Example: Change this prefix to match your data need

# List objects using the selected prefix
object_list = []
for page in paginator.paginate(Bucket='flatfiles', Prefix=prefix):
    for obj in page['Contents']:
        if IS_ALL_DATA:
            object_list.append(obj["Key"])
        else:
            if obj["Key"].split("/")[1] == DATA_TYPE:
                object_list.append(obj["Key"])

#%%
# Specify the bucket name
bucket_name = 'flatfiles'
# Specify the S3 object key name
for pt in tqdm.tqdm(range(len(object_list))):
    object_key = object_list[pt]
    # Specify the local file name and path to save the downloaded file
    # This splits the object_key string by '/' and takes the last segment as the file name
    local_file_name = "_".join([object_key.split("/")[1], object_key.split("/")[-1]])
    # This constructs the full local file path
    local_file_path = os.path.join(os.path.dirname(__file__), '/'.join([DATA_TYPE, local_file_name]))
    if os.path.exists(local_file_path):
        continue
    # Download the file
    s3.download_file(bucket_name, object_key, local_file_path)
    if not os.path.exists(local_file_path):
        pt -= 1
    tqdm.tqdm.write(f"Downloaded {object_key} to {local_file_path}.")

#%%



