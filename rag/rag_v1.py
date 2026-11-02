"""
Based on this tutorial
https://huggingface.co/blog/ngxson/make-your-own-rag

Point for Work (Based on My HERE semantic-search experience)
1. Use Fine-Tuned model for embeddings not only pre-trained one, why ? to improve retrieval (this is an isolated problem)
2. Use https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html ?
3. Top N ? or use some kind of advanced thresholding
4. Experiment with larger datasets
5. Visualized High-Dim embeddings ?
6. # TODO Assess retrieval in RAG ????
7. Stability of response, ask question with numerical quantifiable answers like:
    - On average, how long do cats live ?
    - On average, how many children does a cat have over a lifetime
    - On average, how many species for cats
"""

from tqdm import tqdm
import ollama
from loguru import logger
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
VECTOR_DB = []

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in tqdm(VECTOR_DB,"retrieving relevant chunks"):
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]


def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)


def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))
if __name__ == "__main__":
    # ( 1 ) Load & Chunk
    with open('data/cat-facts.txt', 'r') as file:
        dataset = file.readlines() # each line is the chunk
        logger.info(f'Loaded {len(dataset)} entries')
        # Each element in the VECTOR_DB will be a tuple (chunk, embedding)
        # The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
    # ( 2 ) Adding Chunks
    for chunk in tqdm(dataset,desc='Adding Chunks'):
        add_chunk_to_database(chunk)
    logger.info(f'Added {len(VECTOR_DB)} chunks')

    # ( 3 ) Retrieve , iteratively # TODO should be by vector-db/index

    input_question = "How long does cats live usually?"
    retrieved_knowledge = retrieve(input_question)
    # assess retrieved info
    logger.info('Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        logger.info(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''
    logger.info("Asking the question to the model with retrieved chunks as prompt context")
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_question},
        ],
        stream=True,
    )
    response_chunks = []
    for chunk in stream:
        response_chunks.append(chunk['message']['content'])
    logger.info(f"Response = {''.join(response_chunks)}")


