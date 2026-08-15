# 📰 Fake News Generator & Detector

An educational NLP application that combines **fake-news detection** with **synthetic news generation**. The web interface is built with Streamlit and the models are loaded from Hugging Face.

> ⚠️ Generated text is synthetic. Do not present it as real news. The detector is a machine-learning classifier, not a replacement for professional fact-checking.

## Features

- 🔍 Fake/real news classification using a fine-tuned BERT Tiny model
- ✍️ Synthetic news generation using GPT-2/DistilGPT-2
- 🌐 Streamlit web interface
- ⚡ Models are cached so they are not reloaded on every interaction
- ☁️ Ready for Streamlit Community Cloud deployment

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- BERT Tiny
- GPT-2 family

## Run Locally

```bash
git clone https://github.com/ankit-9335/fake-news-generator-and-detector.git
cd fake-news-generator-and-detector
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the Hugging Face models, so startup can take a little longer.

## Project Structure

```text
fake-news-generator-and-detector/
├── app.py
├── Fake_News_Generator_&_Detector.ipynb
├── Project Report.docx
├── requirements.txt
└── README.md
```

## Model

The detector uses `mrm8488/bert-tiny-finetuned-fake-news-detection`, a BERT Tiny model fine-tuned for fake-news detection. citehttps://huggingface.co/mrm8488/bert-tiny-finetuned-fake-news-detection

## Deployment

This repository is configured for **Streamlit Community Cloud**. Select `app.py` as the application entry point and use the repository's `requirements.txt` during deployment.

## Internship Context

The project was developed as part of a Generative AI internship and explores how generative AI can create synthetic news while NLP models can classify news content. The accompanying project report describes the original IBM internship project and its educational purpose.

## Author

**Ankit Singh**  
B.Tech Computer Science Engineering
