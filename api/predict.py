"""Vercel serverless function: POST /api/predict  (pure Python, no third-party packages)."""
import json
import math
import os
from http.server import BaseHTTPRequestHandler

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.json')


def load_model():
    with open(MODEL_PATH) as f:
        return json.load(f)


def predict(model, values):
    z = [(v - m) / s for v, m, s in zip(values, model['mean'], model['scale'])]
    views = model['linear']['intercept'] + sum(c * x for c, x in zip(model['linear']['coef'], z))
    logit = model['logistic']['intercept'] + sum(c * x for c, x in zip(model['logistic']['coef'], z))
    prob = 1.0 / (1.0 + math.exp(-max(min(logit, 500), -500)))
    return {
        'predicted_views': max(0, round(views)),
        'viral_probability': round(prob, 4),
        'is_viral': prob >= 0.5,
        'viral_threshold_views': round(model['viral_threshold']),
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            model = load_model()
            self._send(200, {'status': 'ok', 'features': model['features'], 'metrics': model['metrics']})
        except FileNotFoundError:
            self._send(500, {'error': 'model.json missing - run train_export.py before deploying'})

    def do_POST(self):
        try:
            model = load_model()
        except FileNotFoundError:
            return self._send(500, {'error': 'model.json missing - run train_export.py before deploying'})
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length) or b'{}')
            values = [float(data[f]) for f in model['features']]
            if any(v < 0 or math.isnan(v) for v in values):
                raise ValueError('values must be non-negative numbers')
        except KeyError as e:
            return self._send(400, {'error': f'missing field: {e.args[0]}'})
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self._send(400, {'error': f'invalid input: {e}'})
        self._send(200, predict(model, values))
