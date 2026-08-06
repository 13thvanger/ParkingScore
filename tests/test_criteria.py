from parking_score.criteria import load_criteria


def test_load_criteria_strips_comments_and_list_prefixes(tmp_path) -> None:
    path = tmp_path / "criteria.txt"
    path.write_text("# comment\n- first\n2. second\n\n\u2022 third\n", encoding="utf-8")

    criteria = load_criteria(path)

    assert criteria.items == ("first", "second", "third")
    assert len(criteria.content_hash) == 64
