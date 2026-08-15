# NewsMorph detector fine-tuning

This directory contains the actual fine-tuning pipeline. The previous notebook loaded `bert-base-uncased` for inference but did not fine-tune it; this pipeline fixes that by training a DistilBERT sequence classifier on WELFake.

## Dataset

WELFake has 72,134 articles: 35,028 real and 37,106 fake. Its labels are `0=fake` and `1=real`.

## Run

Use Google Colab with a GPU (T4 is a practical choice):

```bash
pip install -r training/requirements-train.txt
python training/train_detector.py
```

The script performs duplicate removal, stratified 80/10/10 train/validation/test splitting, title+body preprocessing, tokenization, fine-tuning, early stopping, best-checkpoint selection by F1, and held-out test evaluation.

## After training

Upload the generated `newsmorph-fake-news-detector/` directory to a Hugging Face model repository. Set the Streamlit environment variable `NEWSMORPH_MODEL_ID` to that repository, for example:

`ankit-9335/newsmorph-fake-news-detector`

Do not commit model weights into the GitHub source repository; keep them on the Hugging Face Hub.

## Important

This is a statistical text classifier, not a live fact-checker. A high model probability does not prove a real-world claim is true.
