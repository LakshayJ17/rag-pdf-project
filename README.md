# AskMyPDF

Chat with your PDF using OpenAI and Qdrant, right in your browser!

---

## Features

- Upload any PDF and ask questions about its content.
- First 3 questions are free; continue by entering your own OpenAI API key.
- Fast, accurate answers using GPT-4.1 and Qdrant Cloud.

---

## Try it Online

You can try the app online here: [https://ask-mypdf.streamlit.app/](https://ask-mypdf.streamlit.app/)

---

## How to Run Locally

1. **Clone this repo**
    ```sh
    git clone https://github.com/LakshayJ17/rag-pdf-project.git
    cd rag-pdf-project
    ```

2. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your `.env` file**
    ```
    OPENAI_API_KEY=your-openai-key
    QDRANT_API_KEY=your-qdrant-key
    ```

4. **Run the app**
    ```sh
    streamlit run main.py
    ```
