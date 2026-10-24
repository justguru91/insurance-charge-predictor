"""Local evaluation app. No external web framework or internet required to run."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import csv
import io
import json
import traceback
import pandas as pd
from src.predict import predict, load_bundle, ROOT

MAX_BYTES = 2_000_000

class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, kind='application/json; charset=utf-8'):
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body, allow_nan=False)
        payload = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        routes = {
            '/': ('web/index.html', 'text/html; charset=utf-8'),
            '/app.js': ('web/app.js', 'text/javascript; charset=utf-8'),
            '/style.css': ('web/style.css', 'text/css; charset=utf-8'),
            '/api/dashboard': ('reports/dashboard.json', 'application/json; charset=utf-8'),
            '/example.csv': ('data/example_profiles.csv', 'text/csv; charset=utf-8'),
            '/figures/actual_vs_predicted.png': ('reports/figures/actual_vs_predicted.png', 'image/png'),
            '/figures/residuals.png': ('reports/figures/residuals.png', 'image/png'),
            '/figures/model_comparison.png': ('reports/figures/model_comparison.png', 'image/png'),
            '/figures/charge_distribution.png': ('reports/figures/charge_distribution.png', 'image/png'),
        }
        if self.path == '/api/health':
            return self.respond(200, {'status': 'ready', 'model': load_bundle()['metrics']['selected_model']})
        item = routes.get(self.path)
        if item is None:
            return self.respond(404, {'error': 'Not found.'})
        path = ROOT / item[0]
        if not path.exists():
            return self.respond(503, {'error': 'A project artifact is missing. Run python src/train.py.'})
        return self.respond(200, path.read_bytes(), item[1])

    def do_POST(self):
        if self.path not in ['/api/predict', '/api/batch']:
            return self.respond(404, {'error': 'Not found.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BYTES:
                return self.respond(413, {'error': 'Provide a request between 1 byte and 2 MB.'})
            text = self.rfile.read(length).decode('utf-8-sig')
            if self.path == '/api/predict':
                payload = json.loads(text)
                if not isinstance(payload, dict) or set(payload) != {'profiles'}:
                    raise ValueError('Expected an object with a profiles list.')
                profiles = payload['profiles']
                if not isinstance(profiles, list) or not 1 <= len(profiles) <= 50 or not all(isinstance(p, dict) for p in profiles):
                    raise ValueError('Provide between 1 and 50 profile objects.')
                frame = pd.DataFrame(profiles)
            else:
                header = next(csv.reader(io.StringIO(text)), [])
                if len(header) != len(set(header)):
                    raise ValueError('Duplicate CSV column names are not allowed.')
                frame = pd.read_csv(io.StringIO(text))
                if len(frame) > 2000:
                    raise ValueError('Upload at most 2,000 rows.')
            results = predict(frame)
            return self.respond(200, {'predictions': results.to_dict(orient='records'), 'model': load_bundle()['metrics']['selected_model'], 'currency': 'USD'})
        except (ValueError, TypeError, UnicodeError, StopIteration) as error:
            return self.respond(400, {'error': str(error) or 'Invalid request.'})
        except Exception:
            traceback.print_exc()
            return self.respond(500, {'error': 'Prediction failed. See the server terminal.'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Insurance Charge Predictor locally.')
    parser.add_argument('--port', type=int, default=8502)
    args = parser.parse_args()
    load_bundle()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Insurance Charge Predictor: http://127.0.0.1:{args.port}', flush=True)
    print('Leave this terminal open. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
