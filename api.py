"""
api.py — TIMDR Battery Predict, lokalne REST API + dashboard
================================================================
Serwer Flask udostepniajacy:
  GET  /                  -> dashboard (static/dashboard.html)
  GET  /api/scenarios     -> lista dostepnych scenariuszy demo (nazwa, opis, sugerowany prog)
  GET  /api/demo          -> syntetyczny zestaw czujnikow (?scenario=<nazwa>, domyslnie normal_cycling)
  POST /api/analyze       -> pelna analiza TIMDR (fuse + twist/trend/anomalies/rhythm + TTF + health)
  GET  /api/health        -> healthcheck samego API (nie mylic z health_score ogniwa)

Uruchomienie: `python api.py` (albo `run.bat` na Windows), potem
http://127.0.0.1:5000 w przegladarce.

UWAGA: `TIMDRBatteryFusion.fuse()` ma STALA sygnature (voltage, current,
temperature, resistance) - w przeciwienstwie do TIMDR-Industrial-Predict
/api/analyze nie przyjmuje tu dowolnej listy czujnikow, tylko dokladnie
te 4 nazwane pola.
"""

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from demo_scenarios import DEFAULT_THRESHOLDS, SCENARIOS, make_demo_data
from timdr_battery_fusion import TIMDRBatteryFusion
from timdr_battery_predict import TIMDRBatteryPredict
from timdr_battery_trigger import BatteryTrigger

app = Flask(__name__, static_folder="static", static_url_path="")

fusion = TIMDRBatteryFusion()
predict = TIMDRBatteryPredict()
# Ta sama instancja fusion/predict co wyzej - BatteryTrigger nie
# duplikuje stanu, tylko odpytuje je jeszcze raz z tym samym
# threshold/window co reszta api_analyze() (deterministyczne funkcje,
# wiec wynik jest identyczny z tw_idx/an_idx/ttf ponizej - trigger
# tylko je priorytetyzuje i mapuje na jedno zdarzenie).
trigger = BatteryTrigger(fusion=fusion, predictor=predict)

REQUIRED_SENSORS = ["voltage", "current", "temperature", "resistance"]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify([
        {"id": name, "description": desc, "default_threshold": DEFAULT_THRESHOLDS[name]}
        for name, desc in SCENARIOS.items()
    ])


@app.route("/api/demo")
def api_demo():
    scenario = request.args.get("scenario", "normal_cycling")
    try:
        t, sensors = make_demo_data(scenario)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "scenario": scenario,
        "default_threshold": DEFAULT_THRESHOLDS.get(scenario, 3.0),
        "t": t.tolist(),
        "sensors": {k: v.tolist() for k, v in sensors.items()},
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Body (JSON):
      t: [..]           - znaczniki czasu
      sensors: {voltage: [..], current: [..], temperature: [..], resistance: [..]}
      threshold: float=3.0    - prog "awarii" dla E(t)
      window: int=60          - okno regresji/health_score

    Zwraca pelny wynik analizy jako JSON.
    """
    body = request.get_json(force=True, silent=True) or {}

    if "t" not in body or "sensors" not in body:
        return jsonify({"error": "wymagane pola: 't' (lista) i 'sensors' (obiekt z voltage/current/temperature/resistance)"}), 400

    missing = [s for s in REQUIRED_SENSORS if s not in body["sensors"]]
    if missing:
        return jsonify({"error": f"brakujace czujniki: {missing}. Wymagane wszystkie 4: {REQUIRED_SENSORS}"}), 400

    try:
        t = np.asarray(body["t"], dtype=float)
        sensor_vals = {s: np.asarray(body["sensors"][s], dtype=float) for s in REQUIRED_SENSORS}
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"niepoprawne dane wejsciowe: {exc}"}), 400

    if len(t) == 0:
        return jsonify({"error": "t nie moze byc puste"}), 400

    threshold = float(body.get("threshold", 3.0))
    window = int(body.get("window", 60))

    try:
        E, Z = fusion.fuse(t, sensor_vals["voltage"], sensor_vals["current"],
                            sensor_vals["temperature"], sensor_vals["resistance"])
        tw_idx, tw_z = fusion.twist(t, E)
        tr_sl, tr_z = fusion.trend(t, E, window=window)
        an_idx, an_z = fusion.anomalies(E)
        periods, r_score = fusion.rhythm(E)
        score = fusion.fusion_score(tw_z, tr_z, an_z, r_score)

        ttf, ttf_lin, ttf_exp = predict.predict_failure(t, E, threshold=threshold, window=window)
        health = predict.health_score(E, threshold=threshold, window=window)

        try:
            trigger_result = trigger.analyze(t, E, threshold=threshold, window=window).as_dict()
        except Exception:  # noqa: BLE001 - trigger jest dodatkiem, nie moze wywalic calej analizy
            trigger_result = None
    except Exception as exc:  # noqa: BLE001 - czytelny blad do dashboardu, nie goly 500
        return jsonify({"error": f"blad analizy: {exc}"}), 400

    def clean(x):
        x = float(x)
        return None if not np.isfinite(x) else x

    return jsonify({
        "t": t.tolist(),
        "E": E.tolist(),
        "sensor_names": REQUIRED_SENSORS,
        "twist_idx": tw_idx.tolist(),
        "trend_slopes": tr_sl.tolist(),
        "anomaly_idx": an_idx.tolist(),
        "rhythm_periods": periods,
        "rhythm_score": float(r_score),
        "fusion_score": float(score),
        "ttf": clean(ttf),
        "ttf_linear": clean(ttf_lin),
        "ttf_exp": clean(ttf_exp),
        "health_score": float(health),
        "threshold": threshold,
        "window": window,
        "trigger": trigger_result,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
