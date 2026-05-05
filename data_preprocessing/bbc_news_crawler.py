import html
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


BBC_FEEDS = {
    "front_page": "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml",
    "world": "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/world/rss.xml",
    "business": "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/business/rss.xml",
    "politics": "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/uk_politics/rss.xml",
    "technology": "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/technology/rss.xml",
}
DEFAULT_FEEDS = ("front_page", "world", "business", "technology")
DEFAULT_USER_AGENT = "CS410CourseProjectCrawler/1.0"
MAX_TEXT_CHARS = 50000


class FeedItem:
    def __init__(self, feed, title, url, summary, published_at):
        self.feed = feed
        self.title = title
        self.url = url
        self.summary = summary
        self.published_at = published_at


class BBCArticleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.in_paragraph = False
        self.in_script = False
        self.script_type = ""
        self.current_text = []
        self.article_paragraphs = []
        self.all_paragraphs = []
        self.script_blocks = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == "article":
            self.article_depth += 1
        elif tag == "p":
            self.in_paragraph = True
            self.current_text = []
        elif tag == "script":
            self.in_script = True
            self.script_type = attrs_dict.get("type", "")
            self.current_text = []

    def handle_endtag(self, tag):
        if tag == "article" and self.article_depth > 0:
            self.article_depth -= 1
        elif tag == "p" and self.in_paragraph:
            text = clean_text(" ".join(self.current_text))
            if text:
                self.all_paragraphs.append(text)
                if self.article_depth > 0:
                    self.article_paragraphs.append(text)
            self.in_paragraph = False
            self.current_text = []
        elif tag == "script" and self.in_script:
            if self.script_type == "application/ld+json":
                self.script_blocks.append("".join(self.current_text))
            self.in_script = False
            self.script_type = ""
            self.current_text = []

    def handle_data(self, data):
        if self.in_paragraph or self.in_script:
            self.current_text.append(data)


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_at_word_boundary(text, max_chars):
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.strip()


def fetch_text(url, user_agent, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def parse_rss_items(feed_name, xml_text):
    root = ET.fromstring(xml_text)
    items = []

    for item in root.findall("./channel/item"):
        title = clean_text(item.findtext("title", ""))
        url = clean_text(item.findtext("link", ""))
        summary = clean_text(item.findtext("description", ""))
        published_at = clean_text(item.findtext("pubDate", ""))

        if title and url:
            items.append(
                FeedItem(
                    feed=feed_name,
                    title=title,
                    url=url,
                    summary=summary,
                    published_at=published_at,
                )
            )

    return items


def iter_jsonld_values(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_jsonld_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_jsonld_values(child)


def article_body_from_jsonld(script_blocks):
    # BBC usually puts a clean articleBody in JSON-LD. Prefer that when it is there
    # because it avoids a lot of navigation/footer text.
    for block in script_blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue

        for obj in iter_jsonld_values(data):
            obj_type = obj.get("@type")
            if isinstance(obj_type, list):
                is_article = any(t in ("Article", "NewsArticle") for t in obj_type)
            else:
                is_article = obj_type in ("Article", "NewsArticle")

            body = obj.get("articleBody")
            if is_article and isinstance(body, str):
                return clean_text(body)

    return ""


def extract_article_text(html_text):
    parser = BBCArticleParser()
    parser.feed(html_text)

    jsonld_body = article_body_from_jsonld(parser.script_blocks)
    if jsonld_body:
        return truncate_at_word_boundary(jsonld_body, MAX_TEXT_CHARS)

    # some BBC pages have a normal <article> tag and some do not
    # keep a paragraph fallback instead of failing on one page template
    paragraphs = parser.article_paragraphs or parser.all_paragraphs
    text = clean_text(" ".join(paragraphs))
    return truncate_at_word_boundary(text, MAX_TEXT_CHARS)


def robots_url_for(url):
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))


def can_fetch(url, user_agent, robots_cache):
    robots_url = robots_url_for(url)
    if robots_url not in robots_cache:
        # only fetch robots.txt once per domain to save time when several RSS links point to the same host
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.read()
        robots_cache[robots_url] = parser

    return robots_cache[robots_url].can_fetch(user_agent, url)


def write_record(fout, item, text):
    record = {
        "source": "bbc",
        "feed": item.feed,
        "url": item.url,
        "title": item.title,
        "published_at": item.published_at,
        "summary": item.summary,
        "text": text,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def crawl_bbc(
    feeds,
    output_path,
    max_articles,
    delay_seconds,
    timeout,
    user_agent,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_urls = set()
    robots_cache = {}
    written = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for feed_name in feeds:
            feed_url = BBC_FEEDS[feed_name]
            feed_xml = fetch_text(feed_url, user_agent, timeout)
            items = parse_rss_items(feed_name, feed_xml)
            print(f"{feed_name}: found {len(items)} RSS items")

            for item in items:
                # the same article can show up in multiple BBC feeds
                if item.url in seen_urls:
                    continue
                seen_urls.add(item.url)

                if written >= max_articles:
                    break

                if not can_fetch(item.url, user_agent, robots_cache):
                    print(f"Skipping disallowed URL: {item.url}")
                    continue

                try:
                    article_html = fetch_text(item.url, user_agent, timeout)
                    text = extract_article_text(article_html)
                except Exception as exc:
                    print(f"Skipping failed URL: {item.url} ({exc})")
                    continue

                if len(text) < 200:
                    print(f"Skipping short article: {item.url}")
                    continue

                write_record(fout, item, text)
                written += 1
                print(f"Wrote {written}: {item.title}")
                time.sleep(delay_seconds)

            if written >= max_articles:
                break

    return written


def main():
    feeds = list(DEFAULT_FEEDS)
    output_path = Path("CS410_project") / "data" / "bbc_news" / "bbc_articles.jsonl"
    max_articles = 50
    delay_seconds = 1.0
    timeout = 20
    user_agent = DEFAULT_USER_AGENT

    written = crawl_bbc(
        feeds,
        output_path,
        max_articles,
        delay_seconds,
        timeout,
        user_agent,
    )
    print(f"Wrote {written} BBC articles to {output_path}")


if __name__ == "__main__":
    main()
