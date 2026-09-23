"""Local web dashboard for the tariff campaign agent (standard library server)."""

import argparse
import json
import math
import traceback
from io import StringIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from agent import Agent
from environment import make_environment
from mock_environment import (
    CHANNELS,
    MAX_TOTAL_CONTACTS,
    TOTAL_BUDGET,
    _mock_fallback,
    _mock_impact_model,
)
from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns, validate_strategy


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_REQUEST_BYTES = 25 * 1024 * 1024
CAMPAIGN_COLUMNS = [
    "campaign_name", "filter_arpu_segment", "filter_data_segment",
    "filter_call_segment", "filter_current_tariff", "target_tariff", "channel",
]
DATASETS = {
    "profile": (ROOT / "customer_profile.csv", {
        "ID_NUMBER", "current_tariff", "arpu_segment", "data_segment",
        "call_segment", "predicted_arpu",
    }),
    "history": (ROOT / "data" / "change_tariff.csv", {
        "ID_NUMBER", "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
        "tariff_plan_code_from", "tariff_plan_code_to",
    }),
    "tariffs": (ROOT / "data" / "dict_tariff.csv", {
        "tariff_plan_code", "price_tariff",
    }),
}


def load_data(payload):
    frames = {}
    for key, (default_path, required) in DATASETS.items():
        uploaded = payload.get(f"{key}_csv")
        if uploaded is not None:
            if not isinstance(uploaded, str) or not uploaded.strip():
                raise ValueError(f"Файл {key}: CSV пуст или имеет неверный формат")
            frame = pd.read_csv(StringIO(uploaded))
        else:
            frame = pd.read_csv(default_path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Файл {key}: отсутствуют колонки {', '.join(sorted(missing))}")
        if frame.empty:
            raise ValueError(f"Файл {key}: таблица пуста")
        frames[key] = frame

    profile, history, tariffs = (frames[k] for k in ("profile", "history", "tariffs"))
    if profile["ID_NUMBER"].isna().any() or profile["ID_NUMBER"].duplicated().any():
        raise ValueError("Профиль: ID_NUMBER должен быть заполнен и уникален")
    if tariffs["tariff_plan_code"].isna().any() or tariffs["tariff_plan_code"].duplicated().any():
        raise ValueError("Справочник: tariff_plan_code должен быть заполнен и уникален")
    for frame, col, label in (
        (profile, "predicted_arpu", "Профиль"),
        (history, "AVG_ARPU_PREV_3M", "История"),
        (history, "AVG_ARPU_NEXT_3M", "История"),
        (tariffs, "price_tariff", "Справочник"),
    ):
        try:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{label}: колонка {col} должна содержать числа") from exc
    if profile["predicted_arpu"].isna().any():
        raise ValueError("Профиль: predicted_arpu не должен содержать пустые значения")
    return frames


def run_one(frames, model, seed):
    profile, history, tariffs = (frames[k] for k in ("profile", "history", "tariffs"))
    env, internals = make_environment(
        customer_profile=profile,
        impact_model=model,
        dict_tariff=tariffs,
        channels=CHANNELS,
        total_budget=TOTAL_BUDGET,
        max_total_contacts=MAX_TOTAL_CONTACTS,
        fallback_predict=_mock_fallback,
        seed=seed,
    )
    final = sanitize_campaigns(Agent(change_tariff=history).act(env), tariffs)[:MAX_CAMPAIGNS]
    final_df = pd.DataFrame(final)
    for col in CAMPAIGN_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = None
    if len(final_df):
        validate_strategy(final_df[CAMPAIGN_COLUMNS], tariffs)

    pilots = internals.executed_pilot_campaigns()
    strategy = pd.DataFrame(pilots + final)
    if strategy.empty:
        raise ValueError("Агент не создал ни пилотов, ни кампаний")
    for col in CAMPAIGN_COLUMNS + ["explicit_ids"]:
        if col not in strategy.columns:
            strategy[col] = None
    result = score_campaigns(
        strategy, profile, model, tariffs, float(profile["predicted_arpu"].sum()),
        _mock_fallback, team_id="dashboard",
    )
    campaigns = []
    for i, detail in enumerate(result["campaigns_detail"]):
        source = pilots[i] if i < len(pilots) else final[i - len(pilots)]
        campaigns.append({
            "name": str(detail["name"]),
            "kind": "Пилот" if i < len(pilots) else "Кампания",
            "segment": source.get("filter_arpu_segment") or "Все",
            "current_tariff": source.get("filter_current_tariff") or "Все",
            "target_tariff": str(source["target_tariff"]),
            "channel": str(detail["channel"]),
            "contacts": int(detail["n_contacts"]),
            "cost": float(detail["cost"]),
            "gross_lift": float(detail["gross_lift"]),
        })
    return {
        "seed": seed,
        "status": result["status"],
        "net_gain": float(result["net_arpu_gain"]),
        "gross_lift": float(result["gross_arpu_lift"]),
        "cost": float(result["total_cost"]),
        "baseline": float(result["baseline_total_arpu"]),
        "growth_pct": float(result["growth_vs_baseline_pct"]),
        "contacts": int(result["total_contacts"]),
        "unique_customers": int(result["unique_customers_targeted"]),
        "coverage_pct": float(result["coverage_pct"]),
        "risk_pct": float(result["risk_score_pct"]),
        "roi": float(result["roi"]) if math.isfinite(result["roi"]) else None,
        "pilots": len(pilots),
        "final_count": len(final),
        "campaigns": campaigns,
        "submission": final_df[CAMPAIGN_COLUMNS].where(pd.notna(final_df), None).to_dict("records"),
    }


def run_dashboard(payload):
    try:
        seed = int(payload.get("seed", 42))
        runs = int(payload.get("runs", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Seed и число прогонов должны быть целыми числами") from exc
    if not 0 <= seed <= 2_000_000_000 or not 1 <= runs <= 30:
        raise ValueError("Seed должен быть от 0 до 2 млрд, число прогонов — от 1 до 30")
    frames = load_data(payload)
    model = _mock_impact_model(frames["history"])
    if model.empty or model["conversion_rate"].isna().all():
        raise ValueError("История не содержит подходящих переходов для построения модели")
    results = [run_one(frames, model, seed + i) for i in range(runs)]
    net_values = pd.Series([r["net_gain"] for r in results])
    return {
        "primary": results[0],
        "runs": [{"seed": r["seed"], "net_gain": r["net_gain"], "status": r["status"]} for r in results],
        "summary": {
            "median": float(net_values.median()),
            "minimum": float(net_values.min()),
            "maximum": float(net_values.max()),
            "positive": int((net_values > 0).sum()),
            "total": runs,
        },
        "data": {
            "customers": len(frames["profile"]),
            "history_rows": len(frames["history"]),
            "tariffs": len(frames["tariffs"]),
            "source": "Загруженные данные" if any(payload.get(f"{k}_csv") is not None for k in DATASETS) else "Данные проекта",
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def send_bytes(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        paths = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
        }
        item = paths.get(self.path)
        if item is None:
            self.send_error(404)
            return
        filename, content_type = item
        self.send_bytes((WEB / filename).read_bytes(), content_type)

    def do_POST(self):
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("Размер запроса должен быть не больше 25 МБ")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Ожидался JSON-объект")
            result = run_dashboard(payload)
            body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8")
        except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8", 400)
        except Exception:
            self.log_error("Unexpected dashboard error: %s", traceback.format_exc())
            body = json.dumps({"error": "Не удалось выполнить прогон. Проверьте формат загруженных CSV."}, ensure_ascii=False).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8", 500)


def main():
    parser = argparse.ArgumentParser(description="Tariff campaign dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
