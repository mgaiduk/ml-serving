import os
import json
import torch

from model import load_model, my_hash

def model_fn(model_dir):
    model = load_model(path=os.path.join(model_dir, "model.pth"))
    model.eval()
    return model

def parse_json(js):
    js = json.loads(js)
    ut = torch.tensor(list(map(lambda x: my_hash(x), js["user_ids"])))
    mt = torch.tensor(list(map(lambda x: my_hash(x), js["media_ids"])))
    return ut, mt

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    user_tensors, media_tensors = parse_json(request_body)
    return (user_tensors, media_tensors)


def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object[0], input_object[1])
    return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)