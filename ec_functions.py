import pandas as pd
import numpy as np
import requests
import json
import pickle
import openai
from transformers import GPT2TokenizerFast
from typing import List, Dict, Tuple
import nltk
from urllib.request import urlopen
from typing import Dict, List, Tuple
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)

# Input OpenAI API key
openai.api_key = "YOUR-API-KEY-HERE"

# Set up constants
max_length = 500
temperature = 0.00
separator = "\n* "

# Completions model and embeddings from OpenAI
completions_model = "text-davinci-003"
completions_params = {
    "model": completions_model,
}

# Set up tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(separator))

# Create embedding for a given text
def get_embeddings(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Compute embeddings for each row in a DataFrame, return a dictionary containing the vector and corresponding row index
def compute_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    return {
        idx: get_embeddings(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }

# Load embeddings and their keys from a CSV file
def load_embeddings(fname: str, file) -> Dict[Tuple[str, str], List[float]]:
    file_df = pd.read_csv(fname, header=0)
    file_df2 = pd.read_csv(file)
    load_df = file_df2.merge(file_df, left_index=True, right_index=True)
    max_dim = max([int(c) for c in load_df.columns if c != "title" and c != "heading" and c != 'content' and c != 'tokens'])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in load_df.iterrows()
    }, load_df

# Calculate vector similarity between two vectors
def vector_similarity(x: List[float], y: List[float]) -> float:
    return np.dot(np.array(x), np.array(y))

# Order document sections by their similarity to a given query
def order_by_similarity(query: str, contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
    query_embedding = get_embeddings(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

# Build prompt
def build_prompt(prompt: str, document_embeddings: dict, construct_df: pd.DataFrame) -> str:
    most_relevant_document_sections = order_by_similarity(prompt, document_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, (_, section_index) in most_relevant_document_sections:
        document_section = construct_df[construct_df.heading == section_index]

        for i, row in document_section.iterrows():
            chosen_sections_len += row.tokens + separator_len
            if chosen_sections_len > 2000:
                break
            chosen_sections.append(separator + row.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

    instructions = """Answer the prompt as truthfully as possible using the provided text, and if the answer is not found within the text below, say "Unfortunately, I don't know this one."\n\nContext:\n"""

    return instructions + "".join(chosen_sections) + "\n\n Prompt: " + prompt + "\n Answer:"

# Get a response using build_prompt
def get_response(
    query: str,
    construct_df: pd.DataFrame,
    document_embeddings: Dict[Tuple[str, str], np.array],
    show_prompt: bool = False, temperature=0, max_length=500
) -> str:
    prompt = build_prompt(
        query,
        document_embeddings,
        construct_df
    )

    if show_prompt:
        print(prompt)
    completions_params['temperature'] = temperature
    completions_params['max_tokens'] = max_length

    # Call OpenAI API to get response
    response = openai.Completion.create(
                prompt=prompt,
                **completions_params
            )

    return response["choices"][0]["text"].strip(" \n")

# Prepare earnings call transcript data
def earnings_transcript(ticker, quarter, year):
    url = requests.get(f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?quarter={quarter}&year={year}&apikey=YOUR-API-KEY-HERE")
        
    # Parse JSON data from URL
    def parse_json(url):
        response = urlopen(url)
        data = response.read().decode("utf-8")
        return json.loads(data)
    parsed_df = parse_json(url.url)
    indexed_parsed_df = parsed_df[0]['content']

    # Splits given text into multiple rows
    def split_text(text, length):
        tokens = word_tokenize(text)
        # Calculate the number of rows needed
        rows = len(tokens) // length + (len(tokens) % length != 0)
        # Split the tokens into rows
        split_tokens = [tokens[i:i+length] for i in range(0, len(tokens), length)]
        split_text = [" ".join(row) for row in split_tokens]
        return split_text

    text = indexed_parsed_df
    length = 100
    result = split_text(text, length)

    # Convert the result to a DataFrame
    result_df = pd.DataFrame(result, columns=["content"])
    result_df['title'] = str(parsed_df[0]['symbol']) + ' Q' + str(parsed_df[0]['quarter']) + ' ' + str(parsed_df[0]['year'])
    result_df['heading'] = str(parsed_df[0]['date'])
    result_df['tokens'] = result_df['content'].str.len() / 4

    # Write the DataFrame to a CSV file
    result_df.to_csv('prepared_ec.csv', index=False)

# Return results of prompt
def execute_search(prompt):
    document_embeddings, new_df = load_embeddings('earnings_embeddings.csv', 'prepared_ec.csv')
    prompt_response = get_response(prompt, new_df, document_embeddings, max_length, temperature)
    st.write(prompt_response)