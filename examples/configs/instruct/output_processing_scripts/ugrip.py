import re 
import string 

GEMMA_OUTPUT_IGNORE_REGEX = re.compile(r"<end_of_turn>")
PARENTHESEIS_OUTPUT_IGNORE_REGEX = re.compile(r"\)")
REASONING_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*### Answer: ")

INTEGER_EXTRACTION_REGEX = re.compile(r"\d+")

def process_output(output: str) -> str:
    o1 = output
    output = GEMMA_OUTPUT_IGNORE_REGEX.sub("", output)
    output = REASONING_OUTPUT_IGNORE_REGEX.sub("", output)
    output = PARENTHESEIS_OUTPUT_IGNORE_REGEX.sub("", output)

    match = INTEGER_EXTRACTION_REGEX.search(output)
    if match:
        print("Processing output:", o1, match.group())
        return str(match.group())  # return the number as a string
    
    print("Processing output:", o1, output)
    return str(output)  # or return the cleaned string if no number is found


"""
    Standin... no processing done but this function needs to be
    called by a PreprocessOutputTarget object 
    (see preprocess_output_target.py)
"""
def process_target(output: str) -> str:
    return str(output)