# A3 - Sanskrit to English Machine Translation
![Machine Translation](https://drive.google.com/file/d/1lGygFyGhxzTnmw6D4dk-vwAyN5_mqO-N/view?usp=sharing)
## Overview
This project builds a Sanskrit-to-English neural machine translation system using a seq2seq GRU encoder-decoder with attention. Two attention mechanisms are trained and compared:
- General attention
- Additive attention

The notebook [A3.ipynb](A3.ipynb) covers dataset preparation, EDA, training, evaluation, and attention visualization. A Flask app in [app.py](app.py) serves the best checkpoint for translation.

## Dataset and Tools
- Dataset: Saamayik Sanskrit-English parallel corpus from Hugging Face (acomquest/Saamayik).
- Tokenization:
  - Sanskrit: Indic NLP Library (indic-nlp-library)
  - English: spaCy en_core_web_sm with a basic_english fallback

## Evaluation Results (Validation)
These values are from the latest run in A3.ipynb:

| Attention | Train Loss | Val Loss | Train PPL | Val PPL | BLEU (val) | Avg Epoch Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| General | 5.792994 | 5.831447 | 327.993524 | 340.851645 | 0.026114 | 1422.853089 |
| Additive | 5.770688 | 5.790739 | 320.758182 | 327.254841 | 0.027325 | 1626.981317 |

Plots (loss curves and attention maps) are generated in A3.ipynb under:
- Validation Loss Plot
- Attention Map Visualization

## Project Structure
- A3.ipynb: end-to-end training, evaluation, and plots
- app.py: Flask inference service
- templates/index.html: UI with neumorphism and Three.js background
- models/: saved checkpoints (additive_best.pt, general_best.pt, etc.)
- vocab_transform.pkl: vocabulary object saved from training

## Run the Flask App
1. Ensure checkpoints exist in models/ (from A3.ipynb training).
2. Install dependencies (from workspace requirements):
   - torch==2.3.1
   - torchtext==0.18.0
   - indic-nlp-library
   - spacy (en_core_web_sm)
   - flask
3. Run:

```bash
python app.py
```

Then open http://127.0.0.1:5000.

## Notes
- The app loads additive_best.pt if present, else general_best.pt.
- Attention map rendering may warn about Devanagari fonts; this does not affect inference.
