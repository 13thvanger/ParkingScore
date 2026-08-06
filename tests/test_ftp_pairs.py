from parking_score.ftp_client import build_pairs
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
