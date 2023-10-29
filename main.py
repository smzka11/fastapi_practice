from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer, BertForSequenceClassification
import torch

# 学習済みモデルの読み込み
model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

# インスタンス化
app = FastAPI()

# 入力するデータの方の定義
class SentimentAnalysis(BaseModel):
    text: str


# トップページ
@app.get('/')
async def index():
    return {'SentimentValue': 'sentiment_value'}

# Postが送信された時の処理
@app.post('/sentiment')
async def sentiment(sentiment_analysis: SentimentAnalysis):
    text = sentiment_analysis.text
    result = nlp(text)
    return {'SentimentValue': result[0]['score']}
