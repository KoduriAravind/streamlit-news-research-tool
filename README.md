# streamlit-news-research-tool
# RockyBot: News Research Tool 📈

RockyBot is a Streamlit application designed to process news article URLs, create embeddings, build a FAISS index, and answer user queries using a language model from Hugging Face. The tool provides answers along with references to the sources used.

## Features

- Load and process news articles from given URLs.
- Split large texts into manageable chunks for processing.
- Create embeddings using Hugging Face models.
- Build a FAISS index for efficient querying.
- Answer user queries with references to the original sources.

## Setup

### Prerequisites

- Python 3.7+
- [Hugging Face API token](https://huggingface.co/settings/tokens)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rockybot-news-research-tool.git
   cd rockybot-news-research-tool
2. python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. pip install -r requirements.txt
4. HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
### usage
1.Run the Streamlit application:
bash
Copy code
streamlit run app.py
2.Open the provided local URL in your web browser.

3.In the sidebar, enter up to three news article URLs you want to process.

4.Click the "Process URLs" button to load and process the data.

5.After processing, enter your query in the text input box and press Enter.

6.The answer to your query along with sources will be displayed on the main page.

