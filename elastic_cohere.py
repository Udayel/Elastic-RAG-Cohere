from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
import os
import streamlit as st
from elasticsearch import Elasticsearch
from typing import List, Tuple, Dict
from langchain.llms.bedrock import Bedrock
import boto3
import botocore
from cohere_sagemaker import Client

#Environment variables
cohere_light_endpoint_name = os.environ["COHERE_LIGHT_ENDPOINT"]
aws_region = os.environ["AWS_REGION"]
cid = os.environ['ES_CLOUD_ID']
cp = os.environ['ES_PASSWORD']
cu = os.environ['ES_USERNAME']
index_name = os.environ['ES_INDEX']
max_tokens=1024
max_context_tokens=4000
safety_margin=5

###cohere start

cohere_package = "cohere-gpt-medium-v1-8-081bb643f4ae3394a249d913abc6085c"

# Mapping for Model Packages
model_package_map = {
    "us-east-1": f"arn:aws:sagemaker:us-east-1:865070037744:model-package/{cohere_package}",
    "us-east-2": f"arn:aws:sagemaker:us-east-2:057799348421:model-package/{cohere_package}",
    "us-west-1": f"arn:aws:sagemaker:us-west-1:382657785993:model-package/{cohere_package}",
    "us-west-2": f"arn:aws:sagemaker:us-west-2:594846645681:model-package/{cohere_package}",
    "ca-central-1": f"arn:aws:sagemaker:ca-central-1:470592106596:model-package/{cohere_package}",
    "eu-central-1": f"arn:aws:sagemaker:eu-central-1:446921602837:model-package/{cohere_package}",
    "eu-west-1": f"arn:aws:sagemaker:eu-west-1:985815980388:model-package/{cohere_package}",
    "eu-west-2": f"arn:aws:sagemaker:eu-west-2:856760150666:model-package/{cohere_package}",
    "eu-west-3": f"arn:aws:sagemaker:eu-west-3:843114510376:model-package/{cohere_package}",
    "eu-north-1": f"arn:aws:sagemaker:eu-north-1:136758871317:model-package/{cohere_package}",
    "ap-southeast-1": f"arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/{cohere_package}",
    "ap-southeast-2": f"arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/{cohere_package}",
    "ap-northeast-2": f"arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/{cohere_package}",
    "ap-northeast-1": f"arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/{cohere_package}",
    "ap-south-1": f"arn:aws:sagemaker:ap-south-1:077584701553:model-package/{cohere_package}",
    "sa-east-1": f"arn:aws:sagemaker:sa-east-1:270155090741:model-package/{cohere_package}",
}

if aws_region not in model_package_map.keys():
    raise Exception(f"Current boto3 session region {aws_region} is not supported.")

model_package_arn = model_package_map[aws_region]

co = Client(region_name=aws_region)
co.connect_to_endpoint(endpoint_name=cohere_light_endpoint_name)

##cohere end


LLM_LIST: List[str] = ["Cohere-light"]


# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    print("Connection is", es)
    return es

# Search ElasticSearch index and return body and URL of the result
def search(query_text):
    es = es_connect(cid, cu, cp)
    index = index_name
    
    '''Using ELSER -
       Search ElasticSearch index and return body and URL of the result'''

    query = {
      "bool": {
        "should": [
          {
            "match": {
              "text": {
                "query": query_text,
                "boost": 0.01
              }
            }
          },
          {
            "text_expansion": {
              "ml.inference.text_expanded.predicted_value": {
                "model_id": ".elser_model_1",
                "model_text": query_text
              }
            }
          }
        ]
      }
    }

    fields = ["text",]

#   index = index_name
    resp = es.search(index=index_name,
                     query=query,
                     fields=fields,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['text'][0]

    url=''

    return body, url
def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])


st.title("Anycompany AI Assistant")

with st.sidebar.expander("⚙️", expanded=True):
    llm_model = st.selectbox(label='Choose Large Language Model', options=LLM_LIST)
    with_context = st.checkbox('With Context')


print("Selected LLM Model is:",llm_model)
print("Selected Context Option is:",with_context)

# Main chat form
with st.form("chat_form"):
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")


# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from Context."

if submit_button:
    resp, url = search(query)
    resp = truncate_text(resp, max_context_tokens - max_tokens - safety_margin)
    prompt_with_context = f"Answer this question: {query}\n using only the information from this Elastic Doc: {resp}\n"

    prompt = query

    if with_context:
        prompt = prompt_with_context
    print('prompt is:',prompt)


    if llm_model == "Cohere-light":
        response = co.generate(prompt=prompt, max_tokens=100, temperature=0.9, return_likelihoods='GENERATION')
        answer = response.generations[0].text
    else:
        answer = "Not available. Please select LLM"

    print("Answer is",answer)

    ####stopping here

    if negResponse in answer:
        st.write(f"AI: {answer.strip()}")
    else:
        st.write(f"AI: {answer.strip()}\n\nDocs: {url}")
