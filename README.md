# Customer Service Agent Performance Evaluation

## Overview
This project evaluates the performance of customer service agents using NLP and machine learning techniques. It involves sentiment analysis, keyword extraction, and model training to rank agents based on their performance metrics.

## Installation
Ensure you have Python and pip installed. Install the required dependencies:

```bash
pip install pandas numpy nltk sklearn vaderSentiment spacy
python -m spacy download en_core_web_sm
```

## Usage
Follow these steps to run the project:

1. **Data Preparation**: Load your dataset and prepare it for analysis.
   
2. **Data Preprocessing**: Clean text data, tokenize sentences, and remove stopwords.
   Example:
   ```python
   import pandas as pd
   import re
   from nltk.corpus import stopwords
   from nltk.tokenize import word_tokenize
   
   ```

3. **Feature Extraction**: Extract features such as sentiment scores, TF-IDF keywords, and named entities.
   
4. **Model Training**: Train a RandomForestClassifier to predict resolution outcomes.
   
5. **Scoring and Ranking Agents**: Calculate composite scores and rank agents based on their performance metrics.
   
6. **Visualization**: Use matplotlib to visualize agent performance scores.

## Example Code
Here's an example of how to train the model and score agents:

```python
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
```

## Contributing
Feel free to contribute to this project by forking the repository and submitting pull requests. Please follow the coding conventions and documentation standards.
