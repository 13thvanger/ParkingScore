from parking_score.ftp_client import build_pairs, parse_unix_list_line
from parking_score.models import RemoteFile


def test_pairs_are_matched_in_same_directory_by_stem() -> None:
    files = [
        RemoteFile("/a/id-1.jpg", 10, "1"),
        RemoteFile("/a/id-1.xml", 20, "1"),
        RemoteFile("/b/id-1.jpg", 30, "1"),
        RemoteFile("/b/id-1.xml", 40, "1"),
        RemoteFile("/a/id-1.txt", 5, "1"),
    ]

    pairs = build_pairs(files, (".jpg",))

    assert [(pair.image.path, pair.xml.path) for pair in pairs] == [
        ("/a/id-1.jpg", "/a/id-1.xml"),
        ("/b/id-1.jpg", "/b/id-1.xml"),
    ]


def test_parse_unix_ftp_list_file() -> None:
    parsed = parse_unix_list_line(
        "-rw-r--r--    1 1000 1000 323151 Jul 23 07:54 image name.jpg"
    )

    assert parsed == (
        "image name.jpg",
        {"type": "file", "size": "323151", "modify": "LIST:Jul:23:07:54"},
    )


def test_parse_unix_ftp_list_directory() -> None:
    parsed = parse_unix_list_line(
        "drwxr-xr-x    2 1000 1000 4096 Aug  7 2025 DozorMA687"
    )

    assert parsed is not None
    assert parsed[0] == "DozorMA687"
    assert parsed[1]["type"] == "dir"
