
# Legal Named Entity Recognition (NER) Project

This repository contains code and instructions for using a legal domain-specific Named Entity Recognition (NER) model. It includes both a baseline pre-trained model and a custom-trained version for improved performance.

---

## 📦 Testing the Baseline Model

### 1. Install the pre-trained baseline model

```bash
pip install -r requirements.txt
```

```bash
pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-3.2.0-py3-none-any.whl
```

### 2. Run the baseline model

```bash
python run_legal_ner.py
```

This script will run the baseline model and display the output in your browser for easy visualization of the extracted entities.

---

## 🛠 Using the Designed Model

Our designed model is located in the `training` folder and includes the best-trained configuration for improved performance.

### Components

- `best_model`: The best-trained version of the custom model.
- `train2.py`: Script to train the model using provided datasets.
- `pred.py`: Script to predict legal entities from documents.
- `evaluate2.py`: Script to evaluate the model and calculate performance metrics.
- `check.py` and `check_label.py`: Scripts to ensure data integrity and validate dataset labels.

---

## 🚀 Steps to Use the Designed Model

### 1. Train the Model

Use `train2.py` to train the model with your dataset:

```bash
python ./training/train2.py
```

### 2. Make Predictions

Use `pred.py` to predict legal entities on new documents:

```bash
python training/pred.py
```

### 3. Evaluate the Model

Run `evaluate2.py` to calculate accuracy, precision, recall, and other performance metrics:

```bash
python ./training/evaluate2.py
```

### 4. Check Data Integrity

Use `check.py` and `check_label.py` to verify the integrity of your dataset:

```bash
python ./training/check.py
python ./training/check_label.py
```

---

## 📚 Repository Reference

This project is based on the open-source work available at the [Legal NER GitHub Repository](https://github.com/Legal-NLP-EkStep/legal_NER?tab=readme-ov-file).

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

```plaintext
MIT License

Copyright (c) 2025 Peter Ng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---
