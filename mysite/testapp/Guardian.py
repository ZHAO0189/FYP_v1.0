import json
import requests


class Guardian:

    baseURL = "http://content.guardianapis.com/search?"
    key = 'bb6155b1-d354-41bd-aec8-e00b8801b2a3'

    def __init__(self, company, start_time, end_time):
        self.company = company
        self.start_time = start_time
        self.end_time = end_time

    def get_news(self):

        news = {}
        news_set = []
        counter = 0

        fuzzy_name = self.company.split()
        if len(fuzzy_name) == 1:
            headline_name = fuzzy_name[0]
        else:
            headline_name = fuzzy_name[1]

        url = self.baseURL + 'api-key=' + self.key + '&order=newest&from-date=' + self.start_time + '&to-date=' + \
                  self.end_time + '&show-blocks=all' + '&q=' + self.company.replace(" ", "+")

        response = requests.request("GET", url)
        data = response.text
        js = json.loads(data)

        res = js['response']
        total = res['total']
        #Can't display momre than 200 results.
        if total > 100:
            total = 100
        url = url + '&page-size=' + str(total)
        print url
        response = requests.request("GET", url)
        data = response.text
        js = json.loads(data)

        res = js['response']
        results = res['results']

        for item in results:
            if item['type'] == 'article':
                news.clear()
                news['title'] = item['webTitle']
                if item['webTitle'].lower().find(headline_name.lower()) < 0:
                    continue
                news['url'] = item['webUrl']
                print(item['webUrl'])
                blocks = item['blocks']
                body = blocks['body']
                element = body[0]
                news['content'] = element['bodyTextSummary']
                news_set.append(news.copy())
                counter = counter + 1

        print(str(counter) + " articles saved.")
        #return json.dumps({self.company: news_set})
        return news_set