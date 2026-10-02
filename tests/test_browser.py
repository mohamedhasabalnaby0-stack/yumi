"""Real Chromium smoke checks; chat/voice responses are local fixtures, not paid APIs."""
import base64
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import urllib.request
import wave

import pytest
from playwright.sync_api import sync_playwright, expect


@pytest.fixture(scope="module")
def server():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "GEMINI_API_KEY": "", "ELEVENLABS_API_KEY": ""}
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8765"], cwd=root, env=env)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=1):
                    break
            except OSError:
                if process.poll() is not None:
                    pytest.fail("Server exited during startup")
                time.sleep(.1)
        else:
            pytest.fail("Server did not start")
        yield "http://127.0.0.1:8765"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def audio_fixture():
    buffer = io.BytesIO()
    pcm = b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 220 * i / 16000))) for i in range(32000))
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_model_controls_chat_and_audio(server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if "THREE.PropertyBinding" in message.text or "THREE.WebGLProgram: Shader Error" in message.text else None)
        requests = []
        def reply(route):
            requests.append(route.request.post_data_json)
            route.fulfill(content_type="application/json", body=json.dumps({
                "reply": "Browser test reply", "emotion": "happy", "gesture": "wave",
                "audio": audio_fixture(), "lip_sync_mode": "aligned-text",
                "mouth_cues": [{"start": 0, "end": 2, "weights": {"aa": .7}}],
            }))
        page.route("**/chat", reply)
        try:
            page.goto(server, wait_until="domcontentloaded")
            expect(page.locator("#boot")).to_be_hidden(timeout=120000)
            expect(page.locator("#stage canvas")).to_be_visible()
            page.get_by_role("button", name="Controls", exact=True).click()
            expect(page.locator("#rig-status")).to_contain_text("Available mouth shapes")
            page.locator("#camera-view").select_option("full")
            page.locator("#gesture").select_option("wave")
            page.get_by_role("button", name="Play gesture", exact=True).click()
            page.locator("#motion").uncheck()
            page.locator("#motion").check()
            page.get_by_role("button", name="Controls", exact=True).click()
            page.locator("#message-input").fill("hello")
            page.locator("#composer").evaluate("form => { form.requestSubmit(); form.requestSubmit(); }")
            expect(page.locator("#messages")).to_contain_text("Browser test reply")
            expect(page.locator("#replay")).to_be_enabled()
            assert len(requests) == 1
            page.get_by_role("button", name="Replay voice", exact=True).click()
            expect(page.locator("#status")).to_have_text("Yumi is speaking")
            page.get_by_role("button", name="Stop voice", exact=True).click()
            expect(page.locator("#status")).to_have_text("Here with you")
            assert not errors, errors
        finally:
            Path("browser-artifacts").mkdir(exist_ok=True)
            page.screenshot(path="browser-artifacts/yumi.png", full_page=True)
            browser.close()
