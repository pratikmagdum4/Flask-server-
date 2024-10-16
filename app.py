from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load models for each task
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering")
sentiment_model = pipeline("sentiment-analysis")






@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data['text']
    
    # Summarize the input text
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return jsonify({"summary": summary})


@app.route('/qa', methods=['POST'])
def answer_question():
    data = request.get_json()
    question = data.get('question')
    context = data.get('context')

    if not context:
        return {"error": "Context cannot be empty"}, 400

    try:
        answer = qa_model(question=question, context=context)['answer']
        return {"answer": answer}, 200
    except Exception as e:
        return {"error": str(e)}, 500



@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data['text']
    
    # Analyze sentiment
    result = sentiment_model(text)[0]
    return jsonify({"label": result['label'], "score": result['score']})


if __name__ == '__main__':
    app.run(debug=True)
