import streamlit as st
import boto3 
import os 
from transformers import AutoTokenizer
import torch  
from transformers import pipeline
 


s3 = boto3.client('s3')
local_path = 's3_download'
s3_prefix = "bert-project-sentiment" 
bucket_name = "bert-project-sentiment" 

def download_dir (local_path, s3_prefix):
    os.makedirs(local_path, exist_ok = True)
    paginator = s3.get_paginator('list_objects_v2') 
    for result in paginator.paginate(Bucket = bucket_name, Prefix= s3_prefix):
        if "Contents" in result:
            for key in result["Contents"]:
                s3_key = key['Key']

                local_file = os.path.join(local_path, s3_key)
                os.makedirs(os.path.dirname(local_file), exist_ok= True) 
                s3.download_file(bucket_name, s3_key, local_file)

st.title("Machine Learning Model Deployment at the Server!!!")
button = st.button("Download Model") 
if button:
    with st.spinner("Downloading.... Please wait!"):
        download_dir(local_path= local_path, s3_prefix= s3_prefix)


# 

text = st.text_area("Enter your Review", "Type.....")
predict_button = st.button ("Predict") 
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')  

classifier = pipeline('text-classification', model= "s3_download/bert-project-sentiment", device= device) 
if predict_button:
    with st.spinner("Predicting....."):
        output = classifier(text)
        st.write(output)
        # st.info(output) 