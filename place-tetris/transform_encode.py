def encode(expression):
    # Added support for parentheses and arithmetic symbols
    replacements = {
        '**': 'r',
        ',': '_',
        '(': '-p',
        ')': 'p-',
        '+': 'pl',
        '-': 'mi',
        '*': 'mu',
        '/': 'di',
        'cos': 'tc',
        'sin': 'ts',
    }
    for old, new in replacements.items():
        expression = expression.replace(old, new)
    return expression

def decode(encoded_expression):
    replacements = {
        'r': '**',
        '_': ',',
        '-p': '(',
        'p-': ')',
        'pl': '+',
        'mi': '-',
        'mu': '*',
        'di': '/',
        'tc': 'cos',
        'ts': 'sin',
    }
    # Decoding should happen in reverse order of encoding
    for new, old in reversed(list(replacements.items())):
        encoded_expression = encoded_expression.replace(new, old)
    return encoded_expression

import numpy as np
if __name__ == "__main__":
    # Encoding
    original_expressions = np.array([
        "x**2,x,1",
        "(x+1)*(x-1)",
        "x**3+x**2+x+1",
        "sin(x)+cos(x)",
        "(x/2)-(3*x)+5"
    ])

    encoded_expressions = [encode(expr) for expr in original_expressions]
    print(encoded_expressions)

    # Decoding
    decoded_expressions = [decode(expr) for expr in encoded_expressions]
    print(decoded_expressions)

