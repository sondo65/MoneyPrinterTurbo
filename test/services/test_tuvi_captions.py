from app.services.tuvi import captions


def test_time_words_spreads_syllables_and_pauses():
    words = captions.time_words(["Bạn", "có", "biết,", "Tứ", "Tượng."], 1.0, 5.0)

    assert [w.text for w in words] == ["Bạn", "có", "biết,", "Tứ", "Tượng."]
    assert words[0].start == 1.0
    assert abs(words[-1].end - 5.0) < 1e-6
    gaps = [b.start - a.start for a, b in zip(words, words[1:])]
    # The comma after "biết," adds a pause before "Tứ".
    assert gaps[2] > gaps[0]


def test_chunk_words_breaks_on_size_and_punctuation():
    words = captions.time_words("Bạn có biết bốn thần thú, đó là Tứ Tượng.".split(), 0, 10)
    chunks = [[captions.display_text(w.text) for w in chunk] for chunk in captions.chunk_words(words)]

    assert chunks == [["BẠN", "CÓ", "BIẾT"], ["BỐN", "THẦN", "THÚ"], ["ĐÓ", "LÀ", "TỨ"], ["TƯỢNG"]]
    assert all(len(" ".join(chunk)) <= captions.MAX_CHARS_PER_CHUNK for chunk in chunks)


def test_build_ass_highlights_each_word_and_positions_per_scene():
    first = captions.time_words(["Bạn", "có", "biết"], 0.0, 1.5)
    cta = captions.time_words(["Tải", "app"], 3.0, 4.0)
    ass = captions.build_ass([(first, captions.CAPTION_Y), (cta, 1400)])

    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    caption_lines = [line for line in dialogue if ",Caption," in line]
    assert len([line for line in dialogue if ",Glow," in line]) == 2
    assert len(caption_lines) == 5
    assert f"{{\\1c{captions.GOLD}}}BẠN" in caption_lines[0]
    assert f"BẠN {{\\1c{captions.GOLD}}}CÓ" in caption_lines[1]
    assert "\\pos(540,1400)" in caption_lines[-1]
    assert "PlayResY: 1920" in ass


def test_tokenize_drops_bare_punctuation():
    assert captions.tokenize("Tứ Tượng — bốn  thần thú !") == ["Tứ", "Tượng", "bốn", "thần", "thú"]
