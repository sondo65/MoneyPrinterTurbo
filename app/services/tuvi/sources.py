import html
import json
import os
import random
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Iterable

import requests
from loguru import logger

from app.config import config
from app.utils import utils

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
MAX_ARTICLE_CHARS = 6000
MIN_BLOCK_CHARS = 60
HISTORY_LIMIT = 500

FALLBACK_TOPICS = [
    "Cung Mệnh trong lá số tử vi nói gì về tính cách của bạn",
    "Cung Quan Lộc: sự nghiệp và con đường công danh",
    "Cung Tài Bạch: cách tử vi nhìn chuyện tiền bạc",
    "Cung Phu Thê: tử vi nói gì về tình duyên và hôn nhân",
    "Cung Tật Ách: lời nhắc về sức khỏe trong tử vi",
    "Cung Thiên Di: vận may khi đi xa và ra ngoài xã hội",
    "Sao Tử Vi: đế tinh đứng đầu 14 chính tinh",
    "Sao Thiên Phủ: kho tàng và sự ổn định",
    "Sao Thái Dương và Thái Âm: ánh sáng ngày và đêm trong lá số",
    "Sao Vũ Khúc: tài tinh và tinh thần kỷ luật",
    "Sao Thất Sát: dũng khí và những bước ngoặt",
    "Sao Phá Quân: phá cũ để xây mới",
    "Sao Tham Lang: đa tài, đa tình và khát vọng",
    "Sao Thiên Cơ: trí tuệ và sự linh hoạt",
    "Đại vận 10 năm là gì và vì sao quan trọng",
    "Tiểu vận từng năm: cách đọc vận hạn trong năm",
    "Năm tuổi, Thái Tuế và cách giữ tâm an",
    "Ngày Hoàng đạo và Hắc đạo: hiểu đúng để chọn việc",
    "Mùng 1 và ngày Rằm: những điều nên làm để đón may mắn",
    "Ngũ hành tương sinh tương khắc trong tử vi",
]


@dataclass
class SourceItem:
    title: str
    text: str
    url: str = ""
    source: str = ""
    published: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return (self.url or self.title).strip().lower()


class _TextExtractor(HTMLParser):
    """Collect visible text split at block-level tags, skipping page chrome."""

    _SKIP_TAGS = {
        "script", "style", "noscript", "nav", "header", "footer", "aside",
        "form", "svg", "iframe", "button", "select",
    }
    _BLOCK_TAGS = {
        "p", "div", "li", "h1", "h2", "h3", "h4", "br", "article", "section",
        "td", "tr", "blockquote",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.meta: dict[str, str] = {}
        self.title = ""
        self._current: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "meta":
            attr_map = {k.lower(): (v or "") for k, v in attrs}
            name = attr_map.get("property") or attr_map.get("name") or ""
            if name in {"og:title", "og:description", "description"}:
                self.meta[name] = attr_map.get("content", "")
            return
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        self._current.append(data)

    def _flush(self):
        text = _normalize_space("".join(self._current))
        self._current = []
        if text:
            self.blocks.append(text)

    def close(self):
        super().close()
        self._flush()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "").replace("\xa0", " ")).strip()


def _strip_tags(text: str) -> str:
    return _normalize_space(re.sub(r"<[^>]+>", " ", html.unescape(text or "")))


def _http_get(url: str) -> requests.Response:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "vi-VN,vi;q=0.9"},
        timeout=REQUEST_TIMEOUT,
        verify=config.app.get("tls_verify", True),
    )
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"
    return response


def extract_article(page_html: str, url: str = "") -> SourceItem:
    """Pull the title and main body text out of an article page."""
    parser = _TextExtractor()
    parser.feed(page_html or "")
    parser.close()

    title = _normalize_space(parser.meta.get("og:title") or parser.title)
    seen = set()
    body_blocks = []
    total = 0
    for block in parser.blocks:
        if len(block) < MIN_BLOCK_CHARS or block in seen:
            continue
        seen.add(block)
        body_blocks.append(block)
        total += len(block)
        if total >= MAX_ARTICLE_CHARS:
            break

    text = "\n".join(body_blocks)
    if not text:
        text = _normalize_space(
            parser.meta.get("og:description") or parser.meta.get("description") or ""
        )
    return SourceItem(title=title, text=text[:MAX_ARTICLE_CHARS], url=url, source=url)


def parse_feed(xml_text: str, feed_url: str = "") -> list[SourceItem]:
    """Parse RSS 2.0 or Atom into SourceItems (title + plain-text summary)."""
    try:
        root = ET.fromstring(xml_text.strip().encode("utf-8"))
    except ET.ParseError as exc:
        logger.warning(f"invalid feed xml: {feed_url}, error: {exc}")
        return []

    items = []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag not in {"item", "entry"}:
            continue
        values = {}
        link = ""
        for child in node:
            child_tag = child.tag.rsplit("}", 1)[-1]
            if child_tag == "link":
                link = (child.text or child.get("href") or "").strip() or link
            else:
                values.setdefault(child_tag, child.text or "")
        title = _strip_tags(values.get("title", ""))
        summary = _strip_tags(
            values.get("description") or values.get("summary") or values.get("encoded") or ""
        )
        if not title:
            continue
        items.append(
            SourceItem(
                title=title,
                text=summary,
                url=link,
                source=feed_url,
                published=_normalize_space(
                    values.get("pubDate") or values.get("published") or values.get("updated") or ""
                ),
            )
        )
    return items


def _looks_like_feed(response: requests.Response, url: str) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    if "xml" in content_type or "rss" in content_type or "atom" in content_type:
        return True
    head = response.text.lstrip()[:200].lower()
    return head.startswith("<?xml") or "<rss" in head or "<feed" in head or url.lower().endswith(".rss")


def fetch_from_url(url: str) -> SourceItem | None:
    """Read one article (or the first item of a feed) from a URL sent by the user."""
    try:
        response = _http_get(url)
    except requests.RequestException as exc:
        logger.warning(f"fetch source failed: {url}, error: {exc}")
        return None

    if _looks_like_feed(response, url):
        items = parse_feed(response.text, url)
        return enrich_item(items[0]) if items else None

    item = extract_article(response.text, url)
    if "facebook.com" in url and len(item.text) < 200:
        logger.warning(
            "facebook pages require login; only og:description could be read, "
            "paste the post text to the bot instead"
        )
    return item if (item.title or item.text) else None


def enrich_item(item: SourceItem) -> SourceItem:
    """Replace a short feed summary with the full article body when reachable."""
    if not item.url or len(item.text) >= 800:
        return item
    try:
        article = extract_article(_http_get(item.url).text, item.url)
    except requests.RequestException as exc:
        logger.warning(f"fetch article failed: {item.url}, error: {exc}")
        return item
    if len(article.text) > len(item.text):
        item.text = article.text
    return item


def fetch_feed_items(urls: Iterable[str]) -> list[SourceItem]:
    items = []
    for url in urls:
        url = (url or "").strip()
        if not url:
            continue
        try:
            response = _http_get(url)
        except requests.RequestException as exc:
            logger.warning(f"fetch source failed: {url}, error: {exc}")
            continue
        if _looks_like_feed(response, url):
            items.extend(parse_feed(response.text, url))
        else:
            article = extract_article(response.text, url)
            if article.title or article.text:
                items.append(article)
    return items


class History:
    """Remember which source items were already turned into videos."""

    _lock = threading.Lock()

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(utils.storage_dir("tuvi", create=True), "history.json")

    def _load(self) -> list[dict]:
        try:
            with open(self.path, encoding="utf-8") as fp:
                data = json.load(fp)
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def used_keys(self) -> set[str]:
        with self._lock:
            return {str(entry.get("key", "")) for entry in self._load()}

    def add(self, item: SourceItem, video_path: str = "") -> None:
        with self._lock:
            entries = self._load()
            entries.append(
                {
                    "key": item.key,
                    "title": item.title,
                    "url": item.url,
                    "video": video_path,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fp:
                json.dump(entries[-HISTORY_LIMIT:], fp, ensure_ascii=False, indent=2)


def get_sources() -> list[str]:
    return [str(url) for url in config.tuvi.get("sources", []) if str(url).strip()]


def pick_daily_item(history: History | None = None, sources: list[str] | None = None) -> SourceItem:
    """Pick the freshest unused item from configured sources, else a fallback topic."""
    history = history or History()
    used = history.used_keys()
    candidates = [
        item for item in fetch_feed_items(sources if sources is not None else get_sources())
        if item.key not in used
    ]
    if candidates:
        # Interleave sources so one busy feed does not dominate every day.
        by_source: dict[str, list[SourceItem]] = {}
        for item in candidates:
            by_source.setdefault(item.source, []).append(item)
        source_order = sorted(by_source)
        start = datetime.now().toordinal() % len(source_order)
        chosen = by_source[source_order[start]][0]
        return enrich_item(chosen)

    unused = [t for t in FALLBACK_TOPICS if t.lower() not in used]
    topic = random.choice(unused or FALLBACK_TOPICS)
    logger.info(f"no fresh source item, fallback topic: {topic}")
    return SourceItem(title=topic, text="", source="fallback")


def item_from_user_input(text: str) -> SourceItem:
    """Interpret /taovideo input: a URL, a pasted post, or a short topic."""
    text = (text or "").strip()
    url_match = re.match(r"^(https?://\S+)\s*(.*)$", text, re.S)
    if url_match:
        item = fetch_from_url(url_match.group(1))
        note = url_match.group(2).strip()
        if item:
            if note:
                item.text = f"{note}\n{item.text}".strip()
            return item
        return SourceItem(title=note or url_match.group(1), text=note, url=url_match.group(1))

    if len(text) > 160:
        first_line = text.splitlines()[0][:120]
        return SourceItem(title=first_line, text=text[:MAX_ARTICLE_CHARS], source="user")
    return SourceItem(title=text, text="", source="user")


def item_to_dict(item: SourceItem) -> dict:
    return asdict(item)
