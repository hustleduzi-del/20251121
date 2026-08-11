import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

def fetch_news():
    # '퇴직연금' 키워드로 뉴스 RSS 수집
    query = urllib.parse.quote("퇴직연금")
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    xml_data = response.read()
    
    root = ET.fromstring(xml_data)
    items = root.findall('.//item')
    
    news_list = []
    for item in items[:10]:  # 최신 뉴스 상위 10개 추출
        title_elem = item.find('title')
        link_elem = item.find('link')
        
        if title_elem is not None and link_elem is not None:
            title = html.unescape(title_elem.text)
            link = link_elem.text
            news_list.append({'title': title, 'link': link})
            
    return news_list

def update_readme(news_list):
    today = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    
    content = "# 📰 퇴직연금 최신 뉴스 요약\n\n"
    content += f"> **최종 업데이트:** {today} (KST 기준)\n\n"
    content += "### 📌 주요 뉴스 Top 10\n\n"
    
    for i, news in enumerate(news_list, 1):
        content += f"{i}. [{news['title']}]({news['link']})\n"
        
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    news = fetch_news()
    update_readme(news)
