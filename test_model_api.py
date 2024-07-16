from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import MaximumSequenceProbability

name = "bigscience/bloomz-560m"
model = AutoModelForCausalLM.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)

wb = WhiteboxModel(model, tok)

inputs = ["When was Albert Einstein born?", "What is the capital of France?"]
tokenized = tok(inputs, return_tensors="pt", padding=True, truncation=True)

ue_args = {
    'estimators': [MaximumSequenceProbability()],
}

base_output, ue_output = wb.generate(**tokenized,
                                     max_length=20,
                                     num_return_sequences=1,
                                     do_sample=True,
                                     uncertainty_args=ue_args)
