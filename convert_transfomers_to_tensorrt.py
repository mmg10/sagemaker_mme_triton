

############################
##### THIS DOES NOT WORK####
############################

import torch
import os
import json
from pathlib import Path

from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
# import torch_tensorrt

import transformers
from transformers.onnx import FeaturesManager


EN_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
GER_MODEL = "oliverguhr/german-sentiment-bert"
feature = "sequence-classification"


####################
###### Pre-REQ #####
####################
# pip install transformers[onnx]


####################
###### ENGLISH #####
####################
en_tokenizer = DistilBertTokenizer.from_pretrained(EN_MODEL)
en_model = DistilBertForSequenceClassification.from_pretrained(EN_MODEL).eval()
en_inputs = en_tokenizer("Hello, my dog is cute", return_tensors="pt")
en_script_input = list(en_inputs.values())


# load config
en_model_kind, en_model_onnx_config = FeaturesManager.check_supported_model_or_raise(en_model, feature=feature)
en_onnx_config = en_model_onnx_config(en_model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=en_tokenizer,
        model=en_model,
        config=en_onnx_config,
        opset=13,
        output=Path("model-en.onnx"),
)


####################
###### GERMAN ######
####################

ger_tokenizer = BertTokenizer.from_pretrained(GER_MODEL)
ger_model = BertForSequenceClassification.from_pretrained(GER_MODEL).eval()
ger_inputs = ger_tokenizer("Das ist gar nicht mal so gut", return_tensors="pt")
# print(ger_inputs.keys())
ger_script_input = list(ger_inputs.values())

# load config
ger_model_kind, ger_model_onnx_config = FeaturesManager.check_supported_model_or_raise(ger_model, feature=feature)
ger_onnx_config = ger_model_onnx_config(ger_model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=ger_tokenizer,
        model=ger_model,
        config=ger_onnx_config,
        opset=13,
        output=Path("model-ger.onnx"),
)