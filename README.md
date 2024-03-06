# LLM-jp Factcheck

## Requirements

- Python: 3.10

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Create a `.env` file in the root directory with the following content:

```
AZURE_OPENAI_API_KEY="xxx"
AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com"
AZURE_OPENAI_API_VERSION="2023-05-15"
```

Then run the following command:

```bash
streamlit run src/pipeline.py
```
