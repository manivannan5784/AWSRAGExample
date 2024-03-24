import gradio as gr
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3
import json
 
host = "" #replace this with the AOSS HOST 
region = ""
service = "aoss"
aws_profile_name=''
aws_region_name=''
embedding_model_name='amazon.titan-embed-text-v1'
aws_genai_model_name='anthropic.claude-3-sonnet-20240229-v1:0'

credentials = boto3.Session(profile_name=aws_profile_name).get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

os_client = OpenSearch(
    hosts = [{"host": host, "port": 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)
 
bedrock = boto3.Session(profile_name=aws_profile_name).client(
 service_name='bedrock-runtime',
 region_name=aws_region_name
)

def generate_context(text):
    body=json.dumps({"inputText": text})
    response = bedrock.invoke_model(body=body, modelId=embedding_model_name, accept='application/json', contentType='application/json')
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    document = {
        "size": 15,
        "_source": {"excludes": ["sg_realestate_data_vector"]},
        "query": {
            "knn": {
                 "sg_realestate_data_vector": {
                     "vector": embedding,
                     "k":15
                 }
            }
        }
    }
    response = os_client.search(
    body = document,
    index = "sgrealestate-index"
    )
    data=response['hits']['hits']
    context = ''
    for item in data:
        context += item['_source']['sg_realestate_raw_data'] + '\n'
    return context

def invoke_llma2(augmented_prompt):
    response = bedrock.invoke_model(
                modelId= aws_genai_model_name,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": augmented_prompt}],
                            }
                        ],
                    }
                ),
            )
    # Process and print the response
    result = json.loads(response.get("body").read())
    output_list = result.get("content", [])

    response_text = ""
    for output in output_list:
        response_text = response_text + output["text"]
    print("**** Final Response Received from Model is:  ",response_text)
    return response_text

    
def build_prompt(message, history):
    context=generate_context(message)
    messages=[]
    prompt=f'Context - {context}\nBased on the above context, answer this question in one liner- {message}'
    print(prompt)
    return invoke_llma2(prompt)   

gr.ChatInterface(
    build_prompt,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    title="Singapore 2022 Real Estate Flat Resale Data Powred By Claude3 - (Dev: Manivannan)",
    theme="soft",
    examples=[
        "Whats the highest price a 4 room flat was sold at Ang Mo Kio in 2022 ? ", 
        ],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()