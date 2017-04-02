from django.shortcuts import render
from .form import SignUpForm
from All import get_bloomberg_news, get_guardian_news, get_reuters_news, news_write
from django.views.decorators.csrf import csrf_exempt
from bs4 import BeautifulSoup
from WordEmbeddings import *
from SentenceExtraction import *
# Create your views here.


def home(request):
    title = 'My Title'
    form = SignUpForm()
    context = {
        "template_title": title,
        "form":form
    }
    return render(request, "home.html",context)


def main(request):
    return render(request,'main.html',{})


@csrf_exempt
def crawling(request):
    # context = {}
    # if request.POST:
    #     article_dict = {}
    #     subject = request.POST["subject"]
    #     date = request.POST["date_range"]
    #     news = get_bloomberg_news(subject,date)
    #
    #     for item in news:
    #         title = BeautifulSoup(item['title']).text
    #         article_dict[title] = item["content"]
    #
    #     if len(article_dict) == 0:
    #         context = {
    #             "dict": "No results."
    #         }
    #     else:
    #         context = {
    #             "dict" : article_dict,
    #         }

    context = {}
    if request.POST:
        news = []
        article_dict = {}
        subject = request.POST["subject"]
        date = request.POST["date_range"]
        source = request.POST["source"]

        if source == "Bloomberg":
            news = get_bloomberg_news(subject, date)
        elif source == "Reuters":
            if date == "-1d":
                news = get_reuters_news(subject, start_date="pastDay")
            elif date == "-1w":
                news = get_reuters_news(subject, start_date="pastWeek")
            elif date == "-1m":
                news = get_reuters_news(subject, start_date="pastMonth")
        elif source == "Guardians":
            if date == "-1d":
                news = get_guardian_news(subject, time_dalta=1)
            elif date == "-1w":
                news = get_guardian_news(subject, time_dalta=7)
            elif date == "-1m":
                news = get_guardian_news(subject, time_dalta=30)

        output = json.dumps({subject: news})
        address = news_write(output,subject,source,date)

        for item in news:
            title = BeautifulSoup(item['title']).text
            article_dict[title] = item["content"]
        if len(article_dict) == 0:
            context = {
                'dict': "No results."
            }
        else:
            context = {
                "dict": article_dict,
            }
        request.session["address"] = address
        request.session["subject"] = subject
        request.session.set_expiry(120)
    return render(request,'crawling.html',context)


@csrf_exempt
def classification(request):
    context = {}
    news = []
    article_dict = {}
    sentences_dict = {}
    corpus_a = []
    corpus_b = []
    titles = []

    if request.POST:

        file = request.POST["file-path"]
        subject = request.POST["subject"]

        data = open(file).read()
        data= json.loads(data)
        news = data[subject]
        for item in news:
            corpus_a.append(clean_str(item["content"]))
            corpus_b.append(item["content"])
            title = BeautifulSoup(item["title"]).text
            titles.append(title)
        predictions, positive, negative, neutral = we_predictions(corpus_a)
        sentences = sentence_extraction(corpus_b, predictions)
        i=0
        for title in titles:
            article_dict[title] = predictions[i]
            sentences_dict[title] = sentences[i]
            i += 1

        if len(article_dict) == 0:
            context = {
                'dict': "No results."
            }
        else:
            context = {
                "dict": article_dict,
                "subject": subject,
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
                "sentences": sentences_dict
            }

    if "address" in request.session and "subject" in request.session:
        data = open(request.session["address"]).read()
        data = json.loads(data)
        news = data[request.session["subject"]]

        for item in news:
            corpus_a.append(clean_str(item["content"]))
            corpus_b.append(item["content"])
            title = BeautifulSoup(item["title"]).text
            titles.append(title)

        predictions, positive, negative, neutral = we_predictions(corpus_a)
        sentences = sentence_extraction(corpus_b,predictions)
        i = 0
        for title in titles:
            article_dict[title] = predictions[i]
            sentences_dict[title] = sentences[i]
            i += 1

        if len(article_dict) == 0:
            context = {
                'dict': "No results."
            }
        else:
            context = {
                "dict": article_dict,
                "subject": request.session["subject"],
                "positive":positive,
                "neutral":neutral,
                "negative":negative,
                "sentences":sentences_dict
            }


    return render(request,'classification.html',context)