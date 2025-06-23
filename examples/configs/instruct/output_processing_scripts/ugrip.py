import re 
import string 

GEMMA_OUTPUT_IGNORE_REGEX = re.compile(r"<end_of_turn>")
REASONING_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*### Answer: ")

def process_output(output: str) -> str:
    output = GEMMA_OUTPUT_IGNORE_REGEX.sub("", output)
    output = REASONING_OUTPUT_IGNORE_REGEX.sub("", output)
    return output

"""
    Standin... no processing done but this function needs to be
    called by a PreprocessOutputTarget object 
    (see preprocess_output_target.py)
"""
def process_target(output: str) -> str:
    return output