import streamlit as st
import pandas as pd
import textwrap
import datetime
from transformers import GPT2TokenizerFast
from ec_functions import *
import matplotlib.pyplot as plt
from PIL import Image
    
# Hide Streamlit branding
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Add title image
image = Image.open("title_image.png")
st.image(image, width=250)

# Add subtitle
with st.container():
    st.markdown(
        "<h3 style=''text-align: left; color: #A9A9A9;'>Analyze earnings calls in less time.</h3>",
        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h5 style='margin-top: 15px; margin-bottom: 15px;'>Step 1: Retrieve and prepare the transcript.</h5>", unsafe_allow_html=True)

    # Create columns to display the inputs
    col1, col2, col3 = st.columns(3)

    # Add the input boxes to the columns
    with col1:
        ticker = st.text_input("Enter Ticker:", placeholder="Ex. AAPL")

    with col2:
        year = st.selectbox("Select Year: ", ["2023", "2022", "2021", "2020"])

    with col3:
        quarter = st.selectbox("Select Quarter:", ["1", "2", "3", "4"])

    # Add the button to retrieve the earnings transcript and calculate embeddings
    if st.button("Submit"):
        with st.spinner('Reading the transcript...'):
            df = earnings_transcript(ticker, quarter, year)
            df = pd.read_csv("prepared_ec.csv")
            context_embeddings = compute_embeddings(df)
            df_embeds = pd.DataFrame(context_embeddings).transpose()
            df_embeds.to_csv('earnings_embeddings.csv', index=False)
            st.success("All set! Proceed to Step 2.")

    # Create an empty line
    st.markdown("&nbsp;\n\n")

    # Add Step 2 title
    st.markdown("<h5 style='margin-top: 15px; margin-bottom: 15px;'>Step 2: Select a sample prompt or enter your own:</h5>", unsafe_allow_html=True)

    # Add sample question buttons
    col4, col5, col6 = st.columns(3)

    # Create a placeholder for the answer
    answer_placeholder = st.empty()

    with col4:
        if st.button('"Please summarize the transcript."'):
            prompt = "Please summarize the transcript."
            document_embeddings, new_df = load_embeddings('earnings_embeddings.csv', 'prepared_ec.csv')
            prompt_response = get_response(prompt, new_df, document_embeddings, max_length, temperature)
            answer_placeholder.write(prompt_response)

    with col5:
        if st.button('"What was the company revenue?"'):
            prompt = "What was the company revenue?"
            document_embeddings, new_df = load_embeddings('earnings_embeddings.csv', 'prepared_ec.csv')
            prompt_response = get_response(prompt, new_df, document_embeddings, max_length, temperature)
            answer_placeholder.write(prompt_response)

    with col6:
        if st.button('"What challenges did the company face?"'):
            prompt = "What challenges did the company face?"
            document_embeddings, new_df = load_embeddings('earnings_embeddings.csv', 'prepared_ec.csv')
            prompt_response = get_response(prompt, new_df, document_embeddings, max_length, temperature)
            answer_placeholder.write(prompt_response)

    # Add placeholder within prompt input
    placeholder = "Type your prompt here..."
    prompt = st.text_input("", key="prompt_input", placeholder=placeholder)
    # Temperature ranges from 0-1 and is a measure of randomness. We want as little randomness as possible.
    temperature = 0.00
    max_length = 500
    if st.button("Generate answer"):
        prompt = prompt
        with st.spinner("Thinking..."):
            document_embeddings, new_df = load_embeddings('earnings_embeddings.csv', 'prepared_ec.csv')
            prompt_response = get_response(prompt, new_df, document_embeddings, max_length, temperature)
            st.write(prompt_response)

    # Create an empty line
    st.markdown("&nbsp;\n\n")

    # Add disclaimer
    st.write("<p style='font-size:10px'>Disclaimer: Not financial advice.</p>",
             unsafe_allow_html=True)