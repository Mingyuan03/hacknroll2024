import requests
import json
import pandas as pd
import nltk
nltk.download("punkt")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
URL = "https://shopee.sg/api/v2/item/get_ratings"
ITEMID, SHOPID = "19043502047", "1358899"
LIMIT = 50
QUERYSTRING = {
    "exclude_filter":"1",
    "filter":"1", # '1' for ratings with comments; '0' for ratings without comments
    "filter_size":"0",
    "flag":"1",
    "fold_filter":"0",
    "itemid":ITEMID,
    "limit":str(LIMIT), # max number of ratings
    "offset":"0", # starting rating's index
    "relevant_reviews":"false",
    "request_source":"2",
    "shopid":SHOPID,
    "tag_filter":"",
    "type":"0",
    "variation_filters":""}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
}
"""Data Tokenization"""
RESPONSE = requests.request("GET", url=URL, headers=HEADERS, params=QUERYSTRING)
def get_reviews_shopee(itemid=ITEMID,shopid=SHOPID,limit=None,limit_per_req=59,offset=0):
    url,querystring,headers,response=URL,QUERYSTRING,HEADERS,RESPONSE
    rating_total = response.json()['data']['item_rating_summary']['rating_total']
    rating_count = response.json()['data']['item_rating_summary']['rating_count']
    rcount_with_context = response.json()['data']['item_rating_summary']['rcount_with_context']
    ratings_list=list()
    if limit is None:limit=rcount_with_context
    pages_to_scrape=limit//limit_per_req+1
    for page in range(pages_to_scrape):
        querystring['offset'] = str(offset)
        response = requests.request("GET", url, headers=headers, params=querystring)
        ratings_list += response.json()['data']['ratings']
        offset += limit_per_req
    ratings_df=pd.json_normalize(ratings_list)
    return ratings_df
df=get_reviews_shopee()
MAX_RATING_COUNT,MIN_RATING_COUNT=max(df["rating_star"]),min(df["rating_star"])
"""Sentiment Analysis"""
# Model 1: Roberta Model
ROBERTA_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
# Model 2: Bert Model
BERT_MODEL = f"nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL)
def roberta_classification(text, debug=False):
    # Run for Roberta Model
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict={'roberta_neg':scores[0],'roberta_neu':scores[1],'roberta_pos':scores[2]}
    if debug:print(text)
    return scores_dict
def bert_classification(text, debug=False):
    # Run for BERT Model
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores_dict = {'bert_neg':scores[0],'bert_neu':scores[1],'bert_pos':scores[2]}
    if debug:print(text)
    return scores_dict
"""Comment Analysis"""
def sumy_summarizer(text,language="english",sentences_count=3):
    """Extraction-based Open-source Text Summarizer"""
    english_text=''.join(list(filter(lambda c:c.isascii(),text)))
    parser=PlaintextParser.from_string(english_text,Tokenizer(language))
    summary=LsaSummarizer()(parser.document,sentences_count)
    return ''.join([str(sentence)for sentence in summary])
def sumy_summarization(rating_num,debug=False):
    # Run for SUMY model on df.
    assert rating_num in range(MIN_RATING_COUNT,MAX_RATING_COUNT+1)
    rating_list=','.join(list(df[df["rating_star"]==rating_num]["comment"]))
    if debug:print(rating_list)
    return sumy_summarizer(rating_list)
print(sumy_summarization(3))