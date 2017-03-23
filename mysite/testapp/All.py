# Main file
from datetime import datetime, timedelta, date
from Bloomberg import Bloomberg
from Reuters import Reuters
from Guardian import Guardian


#Companies: Twitter, Goldman Sachs, Facebook, Google, Apple, Amazon
#People: Donald Trump, Hilary Clinton, Putin
#For Donald Trump, 100 articles will be labeled. the rest 50 articles.

def get_bloomberg_news(entity, start_date):
    bb = Bloomberg(entity, start_date)
    output = bb.get_news()
    return output

def get_reuters_news(entity,start_date):
    #start_time can only be either pastDay, pastWeek, pastMonth or pastYear
    reuters = Reuters(entity, start_date)
    output = reuters.get_news()
    return output

def get_guardian_news(entity,time_dalta):
    #start_date = date.today() - timedelte(30)
    start_date = date.today() - timedelta(time_dalta)
    end_date = date.today()
    print(start_date)
    guardian = Guardian(entity, str(start_date), str(end_date))
    output = guardian.get_news()
    return output

def news_write(news, entity, source, period):
    address = "/Users/zhaozinian/Documents/UNIVERSITY/FYP/NEWS/" + source + "-" + entity + "-" + \
              datetime.utcnow().strftime("%Y-%m-%d") + "-" + period + ".json"
    #address = "/Users/zhaozinian/Documents/UNIVERSITY/FYP/NEWS/" + source + "/" + entity + "/2016-11-08.json"

    file = open(address, "w+")
    file.write(news)
    file.close()
    return address


"""
# Start
company = "Donald Trump"
directory = 'E:\\A1113\\FYP\\NEWS\\Unlabeled Corpus\\'


# Get Bloomberg News
b_news = get_bloomberg_news(company)
news_write(b_news, company, "Bloomberg", period= "7")
print("News from Bloomberg Has Been Saved.")
"""
"""
# Get Reuters News
r_news = get_reuters_news(company)
news_write(r_news,company,"Reuters", period="7")
print("News from Reuters Has Been Saved.")


# Get The Guardian News
guardian_news = get_guardian_news(company)
news_write(guardian_news,company,"Guardian", period="7")
print("News from The Guardian Has Been Saved.")
"""



