"""Expected failures through the real FastAPI WebSocket interface."""
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import create_app


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ,{"DENGERKELAS_MODEL_PATH":"data/checkpoints/deliberately-missing"}):
            cls.app=create_app()

    def test_missing_model_keeps_interface_available(self):
        with TestClient(self.app) as client:
            self.assertFalse(client.get('/step2/health').json()['model_ready'])
            self.assertEqual(client.get('/').status_code,200)
            with client.websocket_connect('/step2/audio') as ws:
                ws.send_json({'type':'start','sample_rate':48000})
                self.assertIn('Model lokal belum siap',ws.receive_json()['error'])

    def test_microphone_failure_message_and_reset(self):
        with TestClient(self.app) as client:
            with client.websocket_connect('/step2/audio') as ws:
                ws.send_json({'type':'capture_error','message':'simulated permission denial'})
                self.assertIn('mikrofon gagal',ws.receive_json()['error'])
                ws.send_json({'type':'reset'})
                result=ws.receive_json()
                self.assertEqual(result['type'],'reset')
                self.assertEqual(result['history'],'')

    def test_audio_without_start_is_rejected(self):
        with TestClient(self.app) as client:
            with client.websocket_connect('/step2/audio') as ws:
                ws.send_bytes(b'\x00'*400)
                self.assertIn('Start a session',ws.receive_json()['error'])


if __name__=='__main__':
    unittest.main(verbosity=2)
