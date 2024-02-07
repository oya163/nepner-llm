from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

# Run the server as 
# uvicorn app:app --reload

# Initialize FastAPI instance
app = FastAPI(title='Deploying a LLM Model with FastAPI')

class Text(BaseModel):
    sentence: str

@app.get("/")
def home():
    return "Hello we are going to deploy LLM in production!!"

@app.post("/predict") 
def prediction(text: Text):
    token_classifier = pipeline(
        "token-classification", model='./model/xlm-roberta-large', aggregation_strategy="simple"
    )
    results = token_classifier(text.sentence)
    ret_val = {}
    for each_entity in results:
        ret_val[each_entity['word']] = each_entity['entity_group']
    return ret_val
