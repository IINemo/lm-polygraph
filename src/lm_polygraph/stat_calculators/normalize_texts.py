def normalize(s: str):
    for t in ['<b>', '<strong>']:
        s = s.split(t)[-1]
    for t in [
        '<|eot_id|>', '<eos>', '<end_of_turn>', '\n', '<|end_of_text|>',
        '</b>', '</strong>', '<|im_end|>', '<|endoftext|>', '</think>'
    ]:
        s = s.split(t)[0]
    if 'Answer: ' in s:
        s = s.split('Answer: ')[-1]
    s = s.strip()
    return s