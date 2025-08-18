import tiktoken

# Load ENV file
MODEL = "gpt-3.5-turbo"

# Add this function to track tokens
def count_tokens(messages, model=MODEL):
    """Count tokens for a list of messages"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = 1  # if there's a name, the role is omitted
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        if hasattr(message, 'content') and message.content:
            num_tokens += len(encoding.encode(str(message.content)))
        if hasattr(message, 'role'):
            num_tokens += len(encoding.encode(message.role))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def count_tokens_from_string(text, model=MODEL):
    """Count tokens in a string"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))
