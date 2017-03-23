import json
from datetime import datetime
import urllib2
from bs4 import BeautifulSoup


class Bloomberg:
    baseURL = "http://www.bloomberg.com/api/search?"

    def __init__(self, company, start_time):
        self.company = company
        self.start_time = start_time

    def get_news(self):

        txt = []
        counter = 0
        #fuzzy_name is used to handle names of persons. when crawling data,
        #only articles that have a headline with a person's last name is crawled.
        fuzzy_name = self.company.split()
        if len(fuzzy_name) == 1:
             headline_name = fuzzy_name[0]
        else :
            headline_name = fuzzy_name[1]

        temp = datetime.utcnow().strftime("%Y-%m-%d") + 'T' + datetime.utcnow().strftime("%H:%M:%S.%f")
        end_time = temp[0:len(temp) - 3] + 'Z'

        # Making a request
        url = Bloomberg.baseURL + "query=" + self.company.replace(" ", "+") + "&startTime=" + self.start_time + "&endTime=" + end_time
        #url = Bloomberg.baseURL + "query=" + urllib.parse.quote_plus(self.company) + "&startTime=" + self.start_time + "&endTime=2016-11-08T15:42:37.640Z"
        request = urllib2.Request(url)
        print url
        # Get response and convert it to JSON format
        res = urllib2.urlopen(request)
        data = res.read().decode('utf-8')
        js = json.loads(data)

        # Process the JSON string
        num = js.get("totalResults")
        print "Number of articles = " + str(num)
        page = int(num/10.0) + 1
        print page

        #each page lists 10 results.
        for page_num in range(page):
            add = url + "&page=" + str(page_num+1)
            print(add)
            try:
                req = urllib2.Request(add)
                response = urllib2.urlopen(req)
                results = response.read().decode('utf-8')
                js_str = json.loads(results)

                js_dict = js_str.get("items")
                data = {}
            except Exception as e:
                continue

            for item in js_dict:
                if item.get("storyType") == "Article" and item.get("headline").lower().find(headline_name.lower()) >= 0:
                    data["url"] = item.get("url")
                    data["title"] = item.get("headline")
                    print(item.get("url"))

                    #Go to each url and retrieve the news content
                    try:
                        request = urllib2.Request(item.get("url"))
                        res = urllib2.urlopen(request)
                        soup = BeautifulSoup(res, 'html.parser')
                        """tag = soup.find('div', {"class": "article-body__content"})

                        if tag == None:
                            print("cannot find article-body_content")
                            tag = soup.find('div', {"class": "body-copy"})
                        if tag == None:
                            print("cannot find body-copy")
                            tag = soup.find('div', {"class": "_31WvjDF17ltgFb1fNB1WqY"})
                        if tag == None:
                            print("cannot find Bloomberg View Class.")
                            print(item.get("url"))
                            data["content"] = "Cannot Retrieve Article from the web page. Do it manually."
                            continue
                    except Exception as e:
                        continue
                        """

                        news = ''
                        # technology
                        tag = soup.select('section.main-column p')
                        if not tag:
                            # market
                            tag = soup.select('div.article-body__content > p')
                        if not tag:
                            # view
                            tag = soup.select('div._31WvjDF17ltgFb1fNB1WqY > p')
                        if not tag:
                            # gadfly
                            tag = soup.select('div.container_1KxJx > p')
                        for article in tag:
                            if article.find(class_='inline-newsletter'):
                                continue
                            text = article.get_text()
                            news = news + text + "\n"

                    except Exception as e:
                        continue
                    if news == '':
                        continue
                    data["content"] = news
                    txt.append(data.copy())

                    counter = counter + 1

        #output = json.dumps({self.company: txt})

        print(str(counter) + " articles saved.")
        return txt

    @staticmethod
    def get_baseurl():
        return Bloomberg.baseURL

    def get_company(self):
        return self.company

    def set_company(self, company):
        self.company = company



