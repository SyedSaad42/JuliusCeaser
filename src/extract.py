#this file will implement the code for llm to call and turn the chunk into schema

def extract_chunk(chunk: Chunk) -> dict:
    """
    Given a Chunk, extract the relevant information and return a dictionary
    that can be used to create a Node in the schema.
    """
    # For now, just return the chunk as a dictionary. In a real implementation,
    # you would call the LLM here to extract the relevant information.
    return chunk.dict()

def llm_extract_chunk(chunk: Chunk) -> dict:
    """
    Given a Chunk, call the LLM to extract the relevant information and return a dictionary
    that can be used to create a Node in the schema.
    """
    # Here you would implement the code to call the LLM and extract the information.
    # For now, just return the chunk as a dictionary.
    return chunk.dict()