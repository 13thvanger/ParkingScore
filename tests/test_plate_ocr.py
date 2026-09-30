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


def test_unreadable_without_confidence():
    result = compare_ocr('{"readable":false,"plate":null}', 'O716MP48', 90)
    assert result['plate_check_status'] == 'uncertain'
    assert result['plate_recognized'] is None


@pytest.mark.parametrize('confidence', [None, True, -1, 101, '95'])
def test_readable_requires_valid_confidence(confidence):
    from parking_score.ai_client import AIError
    with pytest.raises(AIError):
        compare_ocr(json.dumps({'plate':'O716MP48','readable':True,
                               'confidence':confidence}), 'O716MP48', 90)


@pytest.mark.parametrize('case,status', [
    ('ok','mismatch'), ('timeout','error'), ('http','error'),
    ('length','error'), ('no_box','unavailable'), ('unreadable','uncertain'),
])
def test_enrichment_is_independent_and_safe(tmp_path, monkeypatch, case, status):
    from types import SimpleNamespace

    from parking_score.ai_client import AIClient
    from parking_score.models import Assessment
    from parking_score.plate_ocr import enrich_assessment
    metadata = parse_recognition_xml(SAMPLE_XML)
    image = tmp_path / 'input.jpg'
    Image.new('RGB', (1920,1200)).save(image)
    observation = SimpleNamespace(cache_image_path=image, plate=metadata.plate,
        plate_box=None if case == 'no_box' else metadata.plate_box,
        image_width=1920, image_height=1200)
    settings = Settings(ftp_host='unused',ftp_port=21,ftp_user='unused',
        ftp_password='unused',ai_api_key='test',plate_ocr_enabled=True,
        ai_debug_export_enabled=True)
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('FTP forbidden')
    monkeypatch.setattr('parking_score.ai_client.FtpClient', forbidden)
    def handler(request):
        calls.append(request)
        assert metadata.plate not in request.content.decode()
        if case == 'timeout':
            raise httpx.ReadTimeout('secret diagnostic must not be logged')
        content = {'plate':'O716MP49','readable':True,'confidence':95}
        if case == 'unreadable':
            content = {'plate':None,'readable':False}
        return httpx.Response(429 if case == 'http' else 200, json={'choices':[
            {'finish_reason':'length' if case == 'length' else 'stop',
             'message':{'content':json.dumps(content)}}]})
    client = AIClient(settings, httpx.MockTransport(handler))
    slots = []
    monkeypatch.setattr(client, '_wait_for_request_slot', lambda: slots.append(True))
    score = Assessment(75, [], '', '{}')
    try:
        result = enrich_assessment(client, settings, observation, score)
    finally:
        client.close()
    assert result.probability == 75
    assert result.plate_check['plate_check_status'] == status
    assert len(calls) == len(slots) == (0 if case == 'no_box' else 1)
