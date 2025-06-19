import streamlit as st
import boto3 
import os 
from transformers import AutoTokenizer
import torch  
from transformers import pipeline
 


s3 = boto3.client('s3')
local_path = 's3_download'
model_dir = os.path.join(local_path, "bert-project-sentiment")
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
        st.success("Download complete!")


# 

text = st.text_area("Enter your Review", "Type.....")
predict_button = st.button ("Predict") 
device = 0 if torch.cuda.is_available() else -1  
# Predict 
# classifier = pipeline('text-classification', model= "s3_download/bert-project-sentiment", device= device) 
if predict_button:
    if not os.path.exists(os.path.join(model_dir,"config.json")):
        st.error("Model not found. Please download it first.")
    else :
        with st.spinner("Loading model and Predicting....."):
            classifier = pipeline('text-classification', model=model_dir, device=device)
            output = classifier(text)
            st.write(output)
            # st.info(output) 