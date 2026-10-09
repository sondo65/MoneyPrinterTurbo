from types import SimpleNamespace

from app.services.tuvi import sources

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Tử vi</title>
<item>
  <title>T&#7913; T&#432;&#7907;ng l&#224; g&#236;?</title>
  <description><![CDATA[<a href="https://example.com/a"><img src="x.jpg"/></a></br>Bốn thần thú trấn giữ bốn phương.]]></description>
  <link>https://example.com/a</link>
  <pubDate>Fri, 09 Oct 2026 00:00:00 GMT</pubDate>
</item>
<item><title>Sao Thất Sát</title><description>Dũng khí</description><link>https://example.com/b</link></item>
</channel></rss>"""

ARTICLE = """<html><head><title>Trang</title>
<meta property="og:title" content="Tứ Tượng là gì?"></head>
<body><nav>Trang chủ | Tử vi | Phong thủy | Liên hệ với chúng tôi ngay hôm nay nhé các bạn thân mến</nav>
<h1>Tứ Tượng</h1>
<div>Tứ Tượng nghĩa là bốn biểu tượng, chỉ bốn sinh vật thần thoại đại diện cho bốn phương.</div>
<p>Ngắn</p>
<p>Thanh Long tượng trưng cho phương Đông và mùa xuân, mang sinh khí của muôn loài.</p>
<script>var x = "Đoạn script dài không được phép xuất hiện trong nội dung bài viết";</script>
</body></html>"""


def test_parse_feed_reads_items_and_strips_html():
    items = sources.parse_feed(RSS, "https://example.com/rss")

    assert [item.title for item in items] == ["Tứ Tượng là gì?", "Sao Thất Sát"]
    assert items[0].text == "Bốn thần thú trấn giữ bốn phương."
    assert items[0].url == "https://example.com/a"
    assert items[0].source == "https://example.com/rss"


def test_extract_article_keeps_body_and_skips_chrome():
    item = sources.extract_article(ARTICLE, "https://example.com/a")

    assert item.title == "Tứ Tượng là gì?"
    assert "bốn sinh vật thần thoại" in item.text
    assert "Thanh Long" in item.text
    assert "Liên hệ" not in item.text
    assert "script" not in item.text
    assert "Ngắn" not in item.text


def test_history_dedupes_items(tmp_path):
    history = sources.History(str(tmp_path / "history.json"))
    item = sources.SourceItem(title="Tứ Tượng", text="", url="https://example.com/a")
    history.add(item, "/video.mp4")

    assert history.used_keys() == {"https://example.com/a"}


def test_pick_daily_item_skips_used_and_falls_back(tmp_path, monkeypatch):
    history = sources.History(str(tmp_path / "history.json"))
    feed_items = sources.parse_feed(RSS, "https://example.com/rss")
    monkeypatch.setattr(sources, "fetch_feed_items", lambda urls: list(feed_items))
    monkeypatch.setattr(sources, "enrich_item", lambda item: item)

    first = sources.pick_daily_item(history, ["https://example.com/rss"])
    assert first.url == "https://example.com/a"
    history.add(first)
    second = sources.pick_daily_item(history, ["https://example.com/rss"])
    assert second.url == "https://example.com/b"
    history.add(second)

    fallback = sources.pick_daily_item(history, ["https://example.com/rss"])
    assert fallback.source == "fallback"
    assert fallback.title in sources.FALLBACK_TOPICS


def test_item_from_user_input_handles_topic_text_and_url(monkeypatch):
    topic = sources.item_from_user_input("Sao Thất Sát ở cung Mệnh")
    assert topic.title == "Sao Thất Sát ở cung Mệnh" and topic.text == ""

    post = "Bài đăng dài\n" + "nội dung tử vi " * 20
    pasted = sources.item_from_user_input(post)
    assert pasted.title == "Bài đăng dài" and "nội dung tử vi" in pasted.text

    monkeypatch.setattr(
        sources,
        "_http_get",
        lambda url: SimpleNamespace(text=ARTICLE, headers={"Content-Type": "text/html"}),
    )
    linked = sources.item_from_user_input("https://example.com/a ghi chú thêm")
    assert linked.url == "https://example.com/a"
    assert linked.text.startswith("ghi chú thêm")
