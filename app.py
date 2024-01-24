import os
import json
import logging
from transformers import pipeline


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

model_dir = os.getenv('MODEL_DIR', "/mnt/ml/models/")

def lambda_handler(event, context):
    # if event.get("source") == "KEEP_LAMBDA_WARM":
    #     LOGGER.info("No ML work to do. Just staying warm...")
    #     return "Keeping Lambda warm"

    token_classifier = pipeline(
        "token-classification", model=os.path.join(model_dir, 'xlm-roberta-large'), aggregation_strategy="simple"
    )
    results = token_classifier(text=event["text"])
    ret_val = {}
    for each_entity in results:
        ret_val[each_entity['word']] = each_entity['entity_group']

    # Function Return 
    return {
        'statusCode': 200,
        'body': json.dumps(ret_val)
    }
