import hashlib
import json

import pytest

from parking_score.criteria import CriteriaError, load_criteria


def _hash(items: list[dict[str, str]]) -> str:
    normalized = json.dumps(
        items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def test_load_legacy_criteria_strips_comments_and_list_prefixes(tmp_path) -> None:
    path = tmp_path / "criteria.txt"
    path.write_text("# comment\n- first\n2. second\n\n• third\n", encoding="utf-8")

    criteria = load_criteria(path)

    assert criteria.items == ("first", "second", "third")
    assert criteria.schema_version == 0
    assert criteria.content_hash.startswith("sha256:")
    assert [item.id for item in criteria.definitions] == [
        "LEGACY001",
        "LEGACY002",
        "LEGACY003",
    ]


def test_load_v2_criteria_uses_canonical_hash_and_automatic_version(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text(
        "# ignored\n[lawn:L01]  все колёса вне асфальта\n"
        "[evidence:E01] кадр пригоден\n[identity:I01] виден целевой автомобиль\n",
        encoding="utf-8",
    )
    expected_hash = _hash(
        [
            {"category": "lawn", "id": "L01", "text": "все колёса вне асфальта"},
            {"category": "evidence", "id": "E01", "text": "кадр пригоден"},
            {
                "category": "identity",
                "id": "I01",
                "text": "виден целевой автомобиль",
            },
        ]
    )
    criteria = load_criteria(criteria_path)

    assert criteria.schema_version == 2
    assert criteria.version == f"txt-{expected_hash.removeprefix('sha256:')[:12]}"
    assert criteria.content_hash == expected_hash
    assert [item.id for item in criteria.definitions] == ["L01", "E01", "I01"]


def test_group_sections_need_only_one_txt_and_generate_stable_ids(tmp_path) -> None:
    first_path = tmp_path / "criteria-1.txt"
    second_path = tmp_path / "criteria-2.txt"
    first_path.write_text(
        "[lawn]\n- все колёса вне асфальта\n"
        "под автомобилем видна трава\n"
        "[evidence] кадр пригоден\n"
        "[identity]\nвиден целевой автомобиль\n",
        encoding="utf-8",
    )
    second_path.write_text(
        "[identity] виден целевой автомобиль\n"
        "[lawn] все колёса вне асфальта\n"
        "[evidence] кадр пригоден\n",
        encoding="utf-8",
    )

    first = load_criteria(first_path)
    second = load_criteria(second_path)

    assert first.schema_version == 2
    assert first.version == f"txt-{first.content_hash.removeprefix('sha256:')[:12]}"
    assert [item.category for item in first.definitions] == [
        "lawn",
        "lawn",
        "evidence",
        "identity",
    ]
    first_ids = {item.text: item.id for item in first.definitions}
    second_ids = {item.text: item.id for item in second.definitions}
    assert first_ids["все колёса вне асфальта"] == second_ids[
        "все колёса вне асфальта"
    ]
    assert first_ids["кадр пригоден"] == second_ids["кадр пригоден"]
    assert first_ids["виден целевой автомобиль"] == second_ids[
        "виден целевой автомобиль"
    ]


def test_grouped_and_legacy_criteria_cannot_be_mixed(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text(
        "legacy criterion\n[lawn]\ngrouped criterion\n", encoding="utf-8"
    )

    with pytest.raises(CriteriaError, match="cannot mix"):
        load_criteria(criteria_path)


def test_v2_rejects_duplicate_ids(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text(
        "[lawn:L01] first\n[evidence:L01] second\n", encoding="utf-8"
    )

    with pytest.raises(CriteriaError, match="IDs must be unique"):
        load_criteria(criteria_path)
