import base64
import io
import json

import httpx
import pytest
from PIL import Image
from test_xml_parser import SAMPLE_XML

from parking_score.config import Settings
from parking_score.plate_ocr import compare_ocr, prepare_plate_crop, run_trial
from parking_score.xml_parser import parse_recognition_xml


@pytest.mark.parametrize("plate,confidence,status", [
    ("О716МР48", 95, "match"), ("О716МР49", 95, "mismatch"),
    ("О716МР49", 70, "uncertain"), ("О?16МР48", 99, "uncertain"),
    (None, 0, "uncertain"),
])
def test_plate_comparison(plate, confidence, status):
    result = compare_ocr(json.dumps({"plate": plate, "readable": plate is not None,
                                     "confidence": confidence}), "O716MP48", 90)
    assert result["plate_check_status"] == status


def test_ocr_sends_only_crop_and_never_ftp(tmp_path, monkeypatch):
    image, xml = tmp_path / "image.jpg", tmp_path / "image.xml"
    Image.new("RGB", (1920, 1200), "white").save(image)
    xml.write_bytes(SAMPLE_XML)

    def forbidden(*args, **kwargs):
        raise AssertionError("FTP must not be used")

    monkeypatch.setattr("parking_score.ai_client.FtpClient", forbidden)

    def handler(request):
        payload = json.loads(request.content)
        assert "O716MP48" not in request.content.decode()
        assert payload["model"] == "test-model"
        assert len(payload["messages"]) == 2
        url = payload["messages"][1]["content"][1]["image_url"]["url"]
        crop = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        assert crop.size == (103, 30)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps({"plate": "О716МР48", "readable": True,
                                                "confidence": 95})}}]})

    settings = Settings(ftp_host="unused", ftp_port=21, ftp_user="unused",
                        ftp_password="unused", ai_api_key="test", ai_model="test-model",
                        ai_debug_export_enabled=True)
    result = run_trial(settings, image, xml, transport=httpx.MockTransport(handler))
    assert result["plate_check_status"] == "match"


def test_invalid_crop_fails_before_ai(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100)).save(image)
    metadata = parse_recognition_xml(SAMPLE_XML)
    with pytest.raises(ValueError, match="aspect ratio"):
        prepare_plate_crop(image, metadata)
