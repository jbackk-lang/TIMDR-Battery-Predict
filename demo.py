from demo_scenarios import DEFAULT_THRESHOLDS, SCENARIOS, make_demo_data
from timdr_battery_fusion import TIMDRBatteryFusion
from timdr_battery_predict import TIMDRBatteryPredict

fusion = TIMDRBatteryFusion()
predict = TIMDRBatteryPredict()

for name, description in SCENARIOS.items():
    t, s = make_demo_data(name)
    threshold = DEFAULT_THRESHOLDS[name]

    E, Z = fusion.fuse(t, s["voltage"], s["current"], s["temperature"], s["resistance"])
    tw_idx, tw_z = fusion.twist(t, E)
    tr_sl, tr_z = fusion.trend(t, E)
    an_idx, an_z = fusion.anomalies(E)
    periods, r_score = fusion.rhythm(E)
    score = fusion.fusion_score(tw_z, tr_z, an_z, r_score)

    ttf, ttf_lin, ttf_exp = predict.predict_failure(t, E, threshold=threshold)
    health = predict.health_score(E, threshold=threshold)

    print(f"=== {name} ===")
    print(f"  {description}")
    print(f"  Fusion score: {score:.2f}")
    print(f"  Punkty twist: {len(tw_idx)}   Anomalie: {len(an_idx)}   Okresy rytmu: {periods[:5]}")
    ttf_str = "brak (inf) - brak trendu ku awarii" if ttf == float("inf") else f"{ttf:.1f}"
    print(f"  Time-to-failure: {ttf_str}   Health score: {health:.3f}")
    print()
