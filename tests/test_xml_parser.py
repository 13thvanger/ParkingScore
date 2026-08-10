from datetime import UTC

from parking_score.xml_parser import (
    extract_pdop,
    has_required_pdop,
    normalize_plate,
    parse_recognition_xml,
)

SAMPLE_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<RecognitionData>
  <CaptureInfo>
    <Id>0a1b12eb-7dd8-444d-a363-3c96134fc510</Id>
    <Date>2026-07-31T08:44:14.171491Z</Date>
    <Number>O716MP48</Number>
  </CaptureInfo>
  <ImagesInfo>
    <ImageWidth>1920</ImageWidth><ImageHeight>1200</ImageHeight>
    <Position><X1>294</X1><Y1>840</Y1><X2>379</X2><Y2>864</Y2></Position>
  </ImagesInfo>
  <Coordinates><Latitude>52.6395215</Latitude><Longitude>39.6612541667</Longitude><Pdop>1.1</Pdop></Coordinates>
  <Address>\xd0\xb3. \xd0\x9b\xd0\xb8\xd0\xbf\xd0\xb5\xd1\x86\xd0\xba, \xd1\x83\xd0\xbb. \xd0\x9a\xd1\x83\xd1\x82\xd1\x83\xd0\xb7\xd0\xbe\xd0\xb2\xd0\xb0, \xd0\xb4. 1</Address>
  <CameraSerialNumber>02DB0494178</CameraSerialNumber>
</RecognitionData>"""


def test_parse_recognition_xml() -> None:
    metadata = parse_recognition_xml(SAMPLE_XML)

    assert metadata.capture_id == "0a1b12eb-7dd8-444d-a363-3c96134fc510"
    assert metadata.plate == "O716MP48"
    assert metadata.camera == "02DB0494178"
    assert metadata.captured_at.tzinfo == UTC
    assert metadata.plate_box is not None
    assert metadata.plate_box.x1 == 294
    assert metadata.image_width == 1920
    assert metadata.pdop == "1.1"


def test_pdop_filter_requires_value_1_1() -> None:
    assert extract_pdop(SAMPLE_XML) == "1.1"
    assert has_required_pdop("1.1")
    assert has_required_pdop(" 1.10 ")
    assert not has_required_pdop(None)
    assert not has_required_pdop("")
    assert not has_required_pdop("1.2")
    assert not has_required_pdop("invalid")


def test_normalize_visually_equivalent_cyrillic_plate() -> None:
    assert normalize_plate("о 716 мр 48") == "O716MP48"


def test_camera_falls_back_to_serial_number_and_position() -> None:
    xml = SAMPLE_XML.replace(
        b"<CameraSerialNumber>02DB0494178</CameraSerialNumber>",
        b"<SerialNumber>01-AA530</SerialNumber><PositionCamera>1</PositionCamera>",
    )

    metadata = parse_recognition_xml(xml, fallback_camera="/DozorMA687")

    assert metadata.camera == "01-AA530/position-1"


def test_camera_falls_back_to_ftp_directory() -> None:
    xml = SAMPLE_XML.replace(
        b"<CameraSerialNumber>02DB0494178</CameraSerialNumber>", b""
    )

    metadata = parse_recognition_xml(xml, fallback_camera="/DozorMA687")

    assert metadata.camera == "ftp:/DozorMA687"
