from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .model import StartupRiskModel
from .predict import DEFAULT_EXAMPLE, predict_payload
from .train import DEFAULT_MODEL_PATH


def make_handler(model: StartupRiskModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "StartupRiskMVP/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.respond_html(index_html())
            elif path == "/health":
                self.respond_json({"status": "ok"})
            elif path == "/model":
                self.respond_json(model.metadata)
            elif path == "/example":
                self.respond_json(DEFAULT_EXAMPLE)
            else:
                self.respond_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/predict":
                self.respond_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self.respond_json(predict_payload(model, payload))
            except json.JSONDecodeError:
                self.respond_json({"error": "invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - last-resort API guard
                self.respond_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def respond_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def respond_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def index_html() -> str:
    example_json = json.dumps(DEFAULT_EXAMPLE, indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Startup Failure Risk MVP</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #657487;
      --line: #d9e0e8;
      --accent: #0f766e;
      --danger: #b42318;
      --warn: #b7791f;
      --ok: #157347;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 20px;
      align-items: start;
    }}
    header {{
      grid-column: 1 / -1;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    p {{ margin: 0; color: var(--muted); line-height: 1.55; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: #344054;
      font-weight: 650;
    }}
    input, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }}
    textarea {{ min-height: 108px; resize: vertical; }}
    .wide {{ grid-column: 1 / -1; }}
    .actions {{
      margin-top: 14px;
      display: flex;
      gap: 10px;
      justify-content: flex-end;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 750;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }}
    button.secondary {{
      background: #e8eef3;
      color: var(--ink);
    }}
    .score {{
      display: grid;
      gap: 8px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 16px;
    }}
    .probability {{
      font-size: 54px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .badge {{
      width: fit-content;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      background: #e8eef3;
    }}
    .badge.high {{ background: #fde8e6; color: var(--danger); }}
    .badge.medium {{ background: #fff4d6; color: var(--warn); }}
    .badge.low {{ background: #e5f5ec; color: var(--ok); }}
    ul {{ margin: 0; padding-left: 20px; color: var(--muted); line-height: 1.55; }}
    li + li {{ margin-top: 10px; }}
    code {{
      background: #eef2f6;
      border-radius: 4px;
      padding: 1px 4px;
      color: #243447;
    }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; }}
      header {{ align-items: start; flex-direction: column; }}
      .grid {{ grid-template-columns: 1fr; }}
      .wide {{ grid-column: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Startup Failure Risk</h1>
        <p>Enter early company signals and get a baseline failure-risk probability with model-derived risk factors.</p>
      </div>
      <p>POST <code>/predict</code></p>
    </header>
    <section>
      <form id="predict-form" class="grid">
        <label>Company name<input name="company_name" value="Example Startup"></label>
        <label>Industry<input name="industry" value="Ecommerce"></label>
        <label>Product type<input name="product_type" value="Marketplace"></label>
        <label>Country<input name="country" value="USA"></label>
        <label>Funding total USD<input name="funding_total_usd" type="number" value="18000000"></label>
        <label>Funding rounds<input name="funding_rounds" type="number" value="3"></label>
        <label>Founded year<input name="founded_year" type="number" value="2022"></label>
        <label>Operating years<input name="operating_years" type="number" value="3"></label>
        <label>Market score<input name="market_score" type="number" value="44"></label>
        <label>Scalability score<input name="scalability_score" type="number" value="58"></label>
        <label class="wide">Company description<textarea name="company_description">A marketplace using subsidies to grow in a crowded category with weak retention.</textarea></label>
        <label class="wide">Founder statement<textarea name="founder_statement">We are still searching for repeat usage after the launch campaign.</textarea></label>
        <div class="actions wide">
          <button type="button" class="secondary" id="load-example">Load example</button>
          <button type="submit">Predict risk</button>
        </div>
      </form>
    </section>
    <section>
      <div id="result" class="score">
        <span class="badge">Ready</span>
        <div class="probability">--</div>
        <p>Submit the form to run the local model.</p>
      </div>
      <ul id="factors"></ul>
    </section>
  </main>
  <script>
    const example = {example_json};
    const form = document.querySelector("#predict-form");
    const result = document.querySelector("#result");
    const factors = document.querySelector("#factors");

    function formPayload() {{
      const data = new FormData(form);
      const payload = Object.fromEntries(data.entries());
      for (const key of ["funding_total_usd", "funding_rounds", "founded_year", "operating_years", "market_score", "scalability_score"]) {{
        payload[key] = Number(payload[key]);
      }}
      return payload;
    }}

    function setForm(payload) {{
      for (const [key, value] of Object.entries(payload)) {{
        const field = form.elements.namedItem(key);
        if (field) field.value = value;
      }}
    }}

    document.querySelector("#load-example").addEventListener("click", () => setForm(example));
    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      result.innerHTML = "<span class='badge'>Running</span><div class='probability'>...</div><p>Scoring the startup profile.</p>";
      factors.innerHTML = "";
      const response = await fetch("/predict", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(formPayload())
      }});
      const payload = await response.json();
      if (!response.ok) {{
        result.innerHTML = `<span class="badge high">Error</span><div class="probability">--</div><p>${{payload.error || "Request failed"}}</p>`;
        return;
      }}
      const percent = Math.round(payload.risk_probability * 100);
      result.innerHTML = `<span class="badge ${{payload.risk_level}}">${{payload.risk_level}}</span><div class="probability">${{percent}}%</div><p>Estimated probability of failure under the MVP model.</p>`;
      factors.innerHTML = payload.top_risk_factors.map((factor) => `<li><strong>${{factor.signal}}</strong><br><code>${{factor.feature}}</code> impact ${{factor.impact}}</li>`).join("");
    }});
  </script>
</body>
</html>"""


def run_server(host: str, port: int, model_path: str) -> None:
    model = StartupRiskModel.load(model_path)
    server = ThreadingHTTPServer((host, port), make_handler(model))
    print(f"serving http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the startup failure prediction API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()
    run_server(args.host, args.port, args.model)


if __name__ == "__main__":
    main()
