from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from lm_polygraph.utils.manager import estimate_uncertainty
from lm_polygraph.estimators import MaximumTokenProbability, LexicalSimilarity, SemanticEntropy, PointwiseMutualInformation, NumSemSets

API_TOKEN = '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'
MODEL_ID = 'google/t5-small-ssm-nq'

model = BlackboxModel.from_huggingface(hf_api_token=API_TOKEN, hf_model_id=MODEL_ID, model_path = MODEL_ID)
ue_method = LexicalSimilarity()
input_text = "Who is George Bush?"
estimate_uncertainty(model, ue_method, input_text=input_text)

