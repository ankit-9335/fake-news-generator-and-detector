# Fake News Generator and Detector

This project includes:
- A fake news detector using ML (Logistic Regression + TF-IDF)
- A fake news generator using GPT-2
- A Streamlit UI for testing

## Setup

```
pip install -r requirements.txt
python detector/train_model.py
```

## Detect News

```
python detector/detect_fake_news.py
```

## Generate News

```
python generator/generate_fake_news.py
```

## Run Web App

```
streamlit run app.py
```
