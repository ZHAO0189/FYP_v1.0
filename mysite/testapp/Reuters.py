import json
import requests
import urllib2
import re
from bs4 import BeautifulSoup


class Reuters:
    baseURL = "http://www.reuters.com/assets/searchArticleLoadMoreJson"

    def __init__(self, company, start_time):
        self.company = company
        self.start_time = start_time


    def get_news(self):

        fuzzy_name = self.company.split()
        if len(fuzzy_name) == 1:
            headline_name = fuzzy_name[0]
        else:
            headline_name = fuzzy_name[1]

        #Request header
        querystring = {"blob": self.company,
                       "bigOrSmall": "big",
                       "articleWithBlog": "true",
                       "sortBy": "relevance",
                       "dateRange": self.start_time,
                       "numResultsToShow": "10",
                       "pn": "1",
                       "callback": "addMoreNewsResults"
                       }

        response = requests.request("GET", self.baseURL, params=querystring)
        txt = response.text
        num = self.extract_num(txt)
        print "Number of results: " + str(num)

        querystring["numResultsToShow"] = str(num)
        querystring.pop("pn")
        response = requests.request("GET", self.baseURL, params=querystring)
        txt = response.text
        links = self.extract_links(txt)


        #Write the news to a JSONArray and each of the news is itself a JSONObject with keys of "title" and "content"
        txt = []
        dict = {}
        counter = 0
        for link in links:
            dict.clear()
            com_link = 'http://www.reuters.com'+link

            try:
                request = urllib2.Request(com_link)
                res = urllib2.urlopen(request)
                soup = BeautifulSoup(res, 'html.parser')

                head = soup.find('h1', 'article-headline')
                if head.getText().lower(). find(headline_name.lower()) < 0:
                    continue
                print com_link
                tag = soup.find('span', {"id": "article-text"})
                list = tag.find_all("p")

                content = ''

                for item in list:
                    content += item.getText() + "\n"
                if len(content.split())<100:
                    continue
                dict["content"] = content
                dict["title"] = head.getText()
                dict["url"] = com_link
                print(com_link)
                counter = counter + 1
                txt.append(dict.copy())
            except Exception as e:
                continue
        print(str(counter) + " articles saved.")
        #return json.dumps({self.company: txt})
        return txt


    # Extract links for the articles
    def extract_links(self, txt):
        #regex = r'http://www\.reuters\.com/article/[\w]+'
        regex = r'/article/[-\w\d]+'
        match = re.findall(regex, txt)
        return match


    #Extract the number of articles
    def extract_num(self, text):
        regex = r'totalResultNumber:\s[\d]+'
        match = re.findall(regex, text)
        num = re.findall(r'[\d]+', match[0])
        return int(num[0])