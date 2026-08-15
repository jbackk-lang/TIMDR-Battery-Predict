"""
demo_scenarios.py — syntetyczne zestawy danych demo dla TIMDR-Battery-Predict
================================================================================
5 scenariuszy dla ogniwa/pakietu baterii - każdy zaprojektowany tak, by
faktycznie uruchamiał deklarowany detektor (zweryfikowane w
test_demo_scenarios.py, nie tylko założone).

UWAGA WAŻNA 1: `TIMDRBatteryFusion.fuse()` ma STAŁĄ sygnaturę
`fuse(t, voltage, current, temperature, resistance)` - w przeciwieństwie
do `TIMDRIndustrialFusion.fuse(t, sensors)` (dowolna lista czujników z
Industrial-Predict) tutaj ZAWSZE trzeba podać wszystkie 4 kanały. Kanały
FIZYCZNIE NIEZWIĄZANE z danym scenariuszem są ustawione jako idealnie
STAŁE (zero wariancji) - `_mad_z` na stałym sygnale zwraca same zera
(`span == 0` -> `zeros_like`), więc taki kanał matematycznie NIE wnosi
nic do E(t). To uproszczenie (realny czujnik nigdy nie jest idealnie
stały) jest jawnie oznaczone w każdej funkcji poniżej.

UWAGA WAŻNA 2: amplitudy zdarzeń dobrane empirycznie tak, żeby
odpowiadające im wartości E(t) mieściły się w rozsądnym zakresie
(dziesiątki, nie setki) - pierwsza iteracja tego pliku używała różnic
rzędu 100-300 odchyleń MAD, co dawało E(t) do 340 i natychmiastowe
TTF=0/health=0 dla każdego scenariusza zdarzeniowego. Zmniejszono
amplitudy zdarzeń, żeby progi (`DEFAULT_THRESHOLDS`) miały sens
predykcyjny, nie tylko binarny "już po awarii".
"""

import numpy as np

SCENARIOS = {
    "normal_cycling": "Normalna praca — regularne cykle ładowanie/rozładowanie (rytm), brak degradacji",
    "capacity_fade": "Starzenie ogniwa — rosnąca rezystancja wewnętrzna i spadek napięcia pod obciążeniem (trend), TTF liczy się w przód",
    "thermal_event": "Zdarzenie termiczne — nagły skok temperatury/rezystancji i zapadnięcie napięcia (anomalia/twist), potem narastająca temperatura",
    "loose_connection": "Luźne połączenie — powtarzalne, narastające skoki rezystancji od wibracji (rytm + twist)",
    "internal_short": "Zwarcie wewnętrzne — nagły skok prądu i spadek napięcia (anomalia), potem narastająca temperatura (trend)",
}

DEFAULT_THRESHOLDS = {
    "normal_cycling": 8.0,
    "capacity_fade": 20.0,
    "thermal_event": 12.0,
    "loose_connection": 8.0,
    "internal_short": 12.0,
}


def normal_cycling(seed=0, n=280, period=40):
    """Normalna praca: prąd i napięcie oscylują sinusoidalnie w fazie ze
    stanem naładowania (SoC). Temperatura i rezystancja NIEZWIĄZANE z tym
    zjawiskiem (brak degradacji) - ustawione idealnie stałe.

    UWAGA (świadomy wybór, zweryfikowany empirycznie): prąd/napięcie w
    kwadratowym cyklu ładowanie/rozładowanie (stała amplituda w obu
    fazach) dawały E(t) PRAKTYCZNIE STAŁE (std=0.036 na tle mediany 0.95)
    - zero wykrywalnego rytmu, bo `E=sqrt(sum(z**2))` dla dwóch kanałów o
    tej samej amplitudzie w obu fazach cyklu daje stałą normę
    niezależnie od fazy. Z sinusoidalnym przebiegiem amplituda faktycznie
    oscyluje, więc E(t) też oscyluje - ale L2 rektyfikuje znak, więc
    wykryty okres to POŁOWA prawdziwego okresu cyklu (harmoniczna
    podwójnej częstotliwości - ten sam, udokumentowany mechanizm co w
    TIMDR-Industrial-Fusion). Test sprawdza wykrycie okresu
    `period // 2`, nie `period`."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    phase = 2 * np.pi * t / period
    current = 2.0 * np.sin(phase) + rng.normal(0, 0.05, n)
    voltage = 3.7 + 0.15 * np.sin(phase) + rng.normal(0, 0.01, n)
    temperature = np.full(n, 25.0)
    resistance = np.full(n, 0.05)
    return t, {"voltage": voltage, "current": current, "temperature": temperature, "resistance": resistance}


def capacity_fade(seed=0):
    """Starzenie: 300 próbek zdrowej pracy, potem 100 próbek narastającej
    rezystancji wewnętrznej i pogłębiającego się spadku napięcia pod
    obciążeniem (typowy powolny trend degradacji ogniwa). Prąd i
    temperatura idealnie stałe."""
    rng = np.random.default_rng(seed)
    n_healthy, n_degrade = 300, 100
    n = n_healthy + n_degrade
    resistance = np.concatenate([
        rng.normal(0.05, 0.002, n_healthy),
        0.05 + np.linspace(0, 0.05, n_degrade) + rng.normal(0, 0.002, n_degrade),
    ])
    voltage = np.concatenate([
        rng.normal(3.70, 0.01, n_healthy),
        3.70 - np.linspace(0, 0.08, n_degrade) + rng.normal(0, 0.01, n_degrade),
    ])
    current = np.full(n, 1.0)
    temperature = np.full(n, 25.0)
    t = np.arange(n, dtype=float)
    return t, {"voltage": voltage, "current": current, "temperature": temperature, "resistance": resistance}


def thermal_event(seed=0):
    """Zdarzenie termiczne: 250 próbek normalnej pracy, nagły skok
    temperatury/rezystancji i zapadnięcie napięcia w momencie zdarzenia
    (t=250), potem 100 próbek narastającej temperatury (propagacja
    ciepła). Wszystkie 4 kanały fizycznie zaangażowane."""
    rng = np.random.default_rng(seed)
    n_healthy, n_after = 250, 100
    n = n_healthy + n_after
    event = n_healthy

    temp = np.concatenate([rng.normal(30, 1, n_healthy), rng.normal(30, 1, n_after)])
    temp[event:event + 3] += [8.0, 14.0, 6.0]
    temp[event + 3:] += np.linspace(0, 10, n - event - 3)

    resistance = np.concatenate([rng.normal(0.05, 0.002, n_healthy), rng.normal(0.05, 0.002, n_after)])
    resistance[event:event + 3] += [0.02, 0.035, 0.015]

    voltage = np.concatenate([rng.normal(3.7, 0.01, n_healthy), rng.normal(3.7, 0.01, n_after)])
    voltage[event:event + 3] -= [0.15, 0.28, 0.12]

    current = np.concatenate([rng.normal(1.0, 0.05, n_healthy), rng.normal(1.0, 0.05, n_after)])

    t = np.arange(n, dtype=float)
    return t, {"voltage": voltage, "current": current, "temperature": temp, "resistance": resistance}


def loose_connection(seed=0, n=300, period=20):
    """Luźne połączenie: regularne, ostre skoki rezystancji co `period`
    próbek (od wibracji), o amplitudzie rosnącej w czasie. Prąd lekko
    reaguje na skok rezystancji. Napięcie i temperatura idealnie stałe."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    base_r = rng.normal(0.05, 0.001, n)
    spike_mask = (np.arange(n) % period) == 0
    growth = np.linspace(0.004, 0.025, n)
    resistance = base_r.copy()
    resistance[spike_mask] += growth[spike_mask]

    current = np.full(n, 1.0)
    current[spike_mask] -= growth[spike_mask] * 0.5

    voltage = np.full(n, 3.7)
    temperature = np.full(n, 25.0)
    return t, {"voltage": voltage, "current": current, "temperature": temperature, "resistance": resistance}


def internal_short(seed=0):
    """Zwarcie wewnętrzne: 250 próbek normalnej pracy, nagły skok prądu i
    spadek napięcia w momencie zwarcia (t=250), potem 100 próbek
    narastającej temperatury (nagrzewanie od zwarcia). Rezystancja
    idealnie stała."""
    rng = np.random.default_rng(seed)
    n_healthy, n_after = 250, 100
    n = n_healthy + n_after
    event = n_healthy

    current = np.concatenate([rng.normal(1.0, 0.05, n_healthy), rng.normal(1.0, 0.05, n_after)])
    current[event:event + 3] += [0.5, 0.9, 0.4]

    voltage = np.concatenate([rng.normal(3.7, 0.01, n_healthy), rng.normal(3.7, 0.01, n_after)])
    voltage[event:event + 3] -= [0.15, 0.25, 0.12]

    temp = np.concatenate([rng.normal(30, 1, n_healthy), 30 + np.linspace(0, 15, n_after) + rng.normal(0, 1, n_after)])
    resistance = np.full(n, 0.05)

    t = np.arange(n, dtype=float)
    return t, {"voltage": voltage, "current": current, "temperature": temp, "resistance": resistance}


GENERATORS = {
    "normal_cycling": normal_cycling,
    "capacity_fade": capacity_fade,
    "thermal_event": thermal_event,
    "loose_connection": loose_connection,
    "internal_short": internal_short,
}


def make_demo_data(scenario="normal_cycling", seed=0):
    if scenario not in GENERATORS:
        raise ValueError(f"Nieznany scenariusz '{scenario}'. Dostepne: {list(GENERATORS)}")
    return GENERATORS[scenario](seed=seed)
