from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__,template_folder='E:/Sentiment Analyzer/',static_folder='static')
analyzer = SentimentIntensityAnalyzer()

import nltk
nltk.download('vader_lexicon')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/service.html')
def service():
    return render_template('service.html')

@app.route('/login.html')
def login():
    return render_template('login.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')



@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.form.get('user-input')
    sentiment_score = perform_sentiment_analysis(text)
    return jsonify({'sentiment_score': sentiment_score})

def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']

if __name__ == '__main__':
    app.run(debug=True, port= 5000)
