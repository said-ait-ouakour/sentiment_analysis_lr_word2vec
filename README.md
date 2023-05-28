# Sentiment Analysis

This is a command-line sentiment analysis app built with logistic regression and word embedding using Word2Vec.

The app allows users to analyze the sentiment (positive or negative) of a given sentence. It uses a pre-trained logistic regression model that has been trained on a labeled dataset for sentiment analysis. Word2Vec is used for word embedding, which converts words into numerical vectors to capture their semantic meaning.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sid41x4/sentiment_analysis_lr_word2vec.git
```

2. Install the required dependencies. It's recommended to use a virtual environment:

```bash
cd sentiment-analysis-app
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate       # For Windows
pip install -r requirements.txt
```

3. Download the pre-trained Word2Vec model. Place the model file `word2vec.model` in the project directory.
4. Run the app:

```css
python main.py -s "sentence to analyze"

```

## Usage

To analyze the sentiment of a sentence, use the following command:

```css
python main.py -s "sentence to analyze"

```

Replace `"sentence to analyze"` with the actual sentence you want to analyze. The app will output the sentiment prediction (positive or negative) based on the trained model.
