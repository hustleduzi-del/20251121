import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

def fetch_news():
    query = urllib.parse.quote("퇴직연금")
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    xml_data = response.read()
    
    root = ET.fromstring(xml_data)
    items = root.findall('.//item')
    
    news_list = []
    for item in items[:10]:
        title_elem = item.find('title')
        link_elem = item.find('link')
        
        if title_elem is not None and link_elem is not None:
            title = html.unescape(title_elem.text)
            link = link_elem.text
            news_list.append({'title': title, 'link': link})
            
    return news_list

def update_html(news_list):
    today = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    
    # 웹사이트 화면(HTML)과 디자인(CSS)을 코드로 생성합니다.
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>퇴직연금 최신 뉴스</title>
    <style>
        body {{
            font-family: 'Malgun Gothic', '맑은 고딕', dotum, sans-serif;
            background-color: #f4f6f9;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #0046FF;
            text-align: center;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        .update-time {{
            text-align: center;
            color: #888;
            font-size: 0.9em;
            margin-bottom: 40px;
        }}
        .news-container {{
            background: #ffffff;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }}
        .news-item {{
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .news-item:last-child {{
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }}
        .news-item a {{
            text-decoration: none;
            color: #002D9D;
            font-size: 1.15em;
            font-weight: bold;
            display: block;
            transition: color 0.2s ease;
        }}
        .news-item a:hover {{
            color: #0061F0;
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>📰 퇴직연금 최신 뉴스 요약</h1>
    <div class="update-time">최종 업데이트: {today} (KST 기준)</div>
    
    <div class="news-container">
"""
    
    # 수집한 뉴스 개수만큼 반복하며 링크를 만들어 줍니다.
    for news in news_list:
        html_content += f"""
        <div class="news-item">
            <a href="{news['link']}" target="_blank">{news['title']}</a>
        </div>
        """
        
    html_content += """
    </div>
</body>
</html>
"""
    
    # 생성된 내용을 index.html 파일에 덮어씁니다.
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    news = fetch_news()
    update_html(news)
