# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flask==3.1.2",
#     "python-dotenv==1.2.1",
# ]
# ///

"""
uv run scripts/sign_server.py
"""

import subprocess, threading, os, sys, shutil, tempfile, secrets, logging
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file

load_dotenv()

# suppress flask/werkzeug output
logging.getLogger("werkzeug").setLevel(logging.ERROR)

REQUIRED_TOOLS = ["jsign", "cloudflared"]
REQUIRED_ENV = ["TUNNEL_URL", "CF_TUNNEL_TOKEN", "PIV_PIN"]


def check_prerequisites():
    missing_tools = [t for t in REQUIRED_TOOLS if not shutil.which(t)]
    if missing_tools:
        print(f"Missing tools: {', '.join(missing_tools)}")
        print("Install them and make sure they're on PATH.")
        sys.exit(1)

    missing_env = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing_env:
        print(f"Missing env vars: {', '.join(missing_env)}")
        print("Add them to .env or export them.")
        sys.exit(1)


check_prerequisites()

SECRET = os.environ.get("TUNNEL_SECRET") or secrets.token_urlsafe(32)
TUNNEL_URL = os.environ["TUNNEL_URL"]
CF_TOKEN = os.environ["CF_TUNNEL_TOKEN"]
PIV_PIN = os.environ["PIV_PIN"]

app = Flask(__name__)


@app.route("/")
def index():
    print(f"[INFO] health check from {request.remote_addr}")
    return jsonify({"status": "ok"})


@app.route("/sign", methods=["POST"])
def sign():
    if request.headers.get("X-Tunnel-Secret") != SECRET:
        print(f"[DENIED] unauthorized request from {request.remote_addr}")
        return jsonify({"error": "unauthorized"}), 401

    file = request.files.get("file")
    if not file or not file.filename:
        print(f"[ERROR] no file in request from {request.remote_addr}")
        return jsonify({"error": "no file provided"}), 400

    print(f"[SIGN] {file.filename} ({request.content_length} bytes) from {request.remote_addr}")

    with tempfile.TemporaryDirectory() as tmp:
        filepath = Path(tmp) / file.filename
        file.save(filepath)

        result = subprocess.run(
            [
                "jsign",
                "--storetype", "YUBIKEY",
                "--storepass", PIV_PIN,
                "--alias", "X.509 Certificate for Digital Signature",
                "--tsaurl", "http://timestamp.digicert.com",
                str(filepath),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[FAIL] jsign failed: {result.stderr.strip()}")
            return jsonify({
                "error": "signing failed",
                "stderr": result.stderr,
                "stdout": result.stdout,
            }), 500

        print(f"[OK] signed {file.filename}")
        return send_file(filepath, as_attachment=True, download_name=file.filename)


def start_tunnel():
    return subprocess.Popen(
        ["cloudflared", "tunnel", "run", "--token", CF_TOKEN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    tunnel = None
    try:
        threading.Thread(
            target=lambda: app.run(port=8080, use_reloader=False),
            daemon=True,
        ).start()
        print("Starting tunnel...")
        tunnel = start_tunnel()
        print(
            f"\nSign server ready at: {TUNNEL_URL}\n"
            f"\n"
            f"Endpoint: POST /sign (multipart file upload)\n"
            f"\n"
            f"  export TUNNEL_URL={TUNNEL_URL}\n"
            f"  export TUNNEL_SECRET={SECRET}\n"
            f"  curl -X POST {TUNNEL_URL}/sign \\\n"
            f'    -H "X-Tunnel-Secret: $TUNNEL_SECRET" \\\n'
            f"    -F 'file=@main.exe' -o signed.exe\n"
            f"\n"
            f"Press Ctrl+C to stop\n"
        )
        tunnel.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if tunnel:
            tunnel.kill()
            tunnel.wait()
        print("Cleaned up")


if __name__ == "__main__":
    main()
