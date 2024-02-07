import json
import os
from transformers import pipeline

model_dir = os.getenv('MODEL_DIR', "/mnt/ml/models/")

def lambda_handler(event, context):

    # Reading the body to extract the URL and the language 
    body = json.loads(event['body'])
    
    token_classifier = pipeline(
        "token-classification", model=os.path.join(model_dir, 'xlm-roberta-large'), aggregation_strategy="simple"
    )
    results = token_classifier(body.sentence)
    response = {}
    for each_entity in results:
        response[each_entity['word']] = each_entity['entity_group']
    
    # Logging the response in the logs
    print(f"Here is the formated output {response}")
    
    # Function Return 
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
