
# Fiscal AI

This project is an interface for quickly getting the information you need from earnings call transcripts. It takes a query, a company ticker, and a quarter/year pair as input and returns a response that attempts to answer the query based on the requested transcript.
    
## Getting Started

1. Clone this repository to your local machine using:

```bash
git clone https://github.com/samfargo/umich_capstone.git
```
2. Navigate to the root directory of the project.

3. Run this command to install the required dependencies:

```bash
pip install -r requirements.txt
```
### APIs

4. There are two APIs used in this Python project. These APIs enable this project to leverage natural language processing tools and financial data to answer questions related to earnings call transcripts:

    OpenAI API:
    OpenAI provides developers with access to state-of-the-art models and tools for natural language processing. In this project, the OpenAI API is used for text completion and text embeddings. The API key for OpenAI is required to use this API, which is set in the openai.api_key variable. A key can be acquired [here](https://platform.openai.com/account/api-keys).

    Financial Modeling Prep API:
    Financial Modeling Prep (FMP) is a financial data provider that offers a wide range of financial data. In this project, the FMP API is used to download earnings call transcripts for a given company ticker, quarter, and year. The earnings_transcript function uses the FMP API to download the transcript data, which is then processed and used for answering questions related to the earnings call. A key can be acquired [here](https://site.financialmodelingprep.com/developer/docs/api-keys).

5. Run this command from the root directory to lauch the app:

```bash
streamlit run app.py
```
