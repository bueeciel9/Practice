import requests
url = "http://www.kric.go.kr/index.jsp"

html_doc = requests.get(url).text
print(html_doc)


# parsing
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify())

# Manipulation
tab = soup.find("table", {"class":"listtbl_c100"})
print(tab)

tdcols = trs[1].find("td", {"class":"tdcol"})
print(tdcols)

tds = trs[1].find_all("td")
print(tds)

tds[0]
tds[0].text

tds[2].text
tds[3].text


stationpassengers = []
for tr in trs[1:]:
    dic = {}
    tds = tr.find_all("td")
    dic["station"] = tds[0].text
    dic["passengers"] = tds[2].text
    dic['alight'] = tds[3].text
    stationpassengers.append(dic)

print(stationpassengers)


# Open API

# http://data.ex.co.kr/openapi/trafficapi/trafficRegion

import requests
key = "YOUR KEY"
type = "json"
StartUnitCode = '101'
EndUnitCode = '103'

URL = "http://data.ex.co.kr/openapi/trafficapi/trafficRegion"
url = URL + "?key=" + key + "&type=" + type + "&StartUnitCode=" + StartUnitCode + "&EndUnitCode=" + EndUnitCode

response = requests.get(url)
print(response.text)
