# TIMDR-Battery-Predict

Predykcyjne utrzymanie ogniwa/pakietu baterii: fuzja 4 czujników
(napięcie, prąd, temperatura, rezystancja wewnętrzna) w jeden sygnał
"energii stanu" E(t), plus standardowy zestaw detektorów TIMDR (twist,
trend, anomalie, rytm), model degradacji, prognoza czasu do awarii
(TTF) i wynik zdrowia (health_score). Lokalne REST API (Flask) +
dashboard bez zależności od CDN + `run.bat`.

## Status

Kod ze zgłoszenia (`TIMDRBatteryFusion`, `TIMDRBatteryPredict`)
uruchomiony i przetestowany (32/32 testów). **Zweryfikowano empirycznie
i naprawiono 4 błędy** — wszystkie identyczne co do mechanizmu z
wcześniej znalezionymi i naprawionymi błędami w TIMDR-Industrial-Predict
(kod ze zgłoszenia wygląda na skopiowany z wcześniejszej,
przed-poprawkową wersję tamtego modułu). Oryginalne pliki zachowane
jako `timdr_battery_fusion_original.py` / `timdr_battery_predict_original.py`
do porównania.

Po drodze sprawdzona i **odrzucona** została też trzecia, "poprawiona"
wersja kodu (`TIMDRBattery` z metodą "wygaszania" outlierów) zgłoszona w
międzyczasie — testy pokazały, że nie naprawia żadnego z rzeczywistych
problemów, a jeden z nich pogarsza (patrz sekcja "Odrzucona trzecia
wersja" niżej).

## 🐛 Błąd 1: `rhythm()` mylił trend degradacji z okresowością

Oryginalny kod (`timdr_battery_fusion_original.py`) tylko odejmował
średnią (`E - mean(E)`) i zgłaszał **każdy** lag powyżej `power_thresh`,
nie tylko lokalne maksima. Dla sygnału z trendem (typowe dla starzenia
ogniwa - rosnąca rezystancja, spadające napięcie pod obciążeniem)
kolejne próbki są do siebie podobne z powodu gładkiego trendu, nie
cykliczności.

Zweryfikowano: czysty rosnący trend (bez śladu okresowości, z
realistycznym szumem czujnika, n=200) dawał **`rhythm_score=0.989` i 48
"wykrytych okresów"**.

Naprawiono (`timdr_battery_fusion.py`): pełny detrend (nachylenie +
wyraz wolny, nie tylko średnia) przed liczeniem autokorelacji, plus
zgłaszanie tylko lokalnych maksimów powyżej progu. Ten sam test po
poprawce daje `[]`, `0.0`.

## 🐛 Błąd 2: `predict_failure()` zwracał czas absolutny, nie czas do awarii

Oryginalny kod rozwiązywał `E(t)=threshold` względem tej samej osi `t`
co dane wejściowe i zwracał to jako "TTF" — w rzeczywistości zwracał
**współrzędną na osi t**, nie czas od ostatniego pomiaru.

Zweryfikowano dwoma testami:
- **Przesunięcie zegara**: identyczna degradacja, `t` przesunięte o
  +1000, dawała TTF większe o dokładnie **+1000**.
- **Realne znaczniki epoki** (rzędu 1.7e9): TTF wychodziło rzędu
  **miliardów jednostek** zamiast rozsądnej liczby (numeryczne
  niedokondycjonowanie `lstsq` na nieucentrowanym `t`, dokładnie ten sam
  mechanizm co w TIMDR-Industrial-Predict).
- **Brak okna**: regresja liczona na CAŁEJ historii, więc identyczna
  "świeża" degradacja dawała różny TTF w zależności od tego, ile
  zdrowej historii ją poprzedzało.

Naprawiono (`timdr_battery_predict.py`): `degradation_model()` centruje
czas (`t0 = t[window_start]`) i ogranicza dopasowanie do ostatniego
`window` próbek (domyślnie 60); `predict_failure()` odejmuje pozycję
ostatniej próbki (`t_ref`) od momentu przecięcia progu, więc wynik to
naprawdę czas **od teraz**, niezależny od punktu odniesienia zegara i
od długości wcześniejszej historii.

## 🐛 Błąd 3: `health_score()` trwale "zatruty" jednym starym skokiem

Oryginalny kod liczył `max(|MAD-z(E)|)` na **całej** historii E — jeden
stary skok na początku nagrania dawał `health_score=0.0` **na zawsze**,
nawet gdy wszystkie kolejne setki próbek były normalne.

Zweryfikowano: pojedynczy skok w próbce 50 (z 500) dawał
`health_score=0.000`, mimo 450 zdrowych próbek po nim.

Naprawiono: liczymy medianę z **ostatniego okna** (domyślnie 20 próbek)
i porównujemy ją bezpośrednio do tego samego `threshold`, co
`predict_failure` — wynik odzwierciedla aktualny stan i jest spójny z
tym samym progiem awarii. Ten sam test po poprawce daje ok. 0.68
(zdrowy).

## 🐛 Błąd 4 (drobny, ale realny): `fuse()` wymaga wszystkich 4 kanałów naraz

Nie jest to "bug" w sensie błędnej matematyki, ale ograniczenie API
warte odnotowania: `fuse(t, voltage, current, temperature, resistance)`
ma sztywną sygnaturę — w przeciwieństwie do `TIMDRIndustrialFusion.fuse(t,
sensors)` (dowolna lista czujników) tutaj zawsze trzeba podać wszystkie
4 kanały. Dla scenariuszy, gdzie tylko 1-2 czujniki są fizycznie
związane ze zjawiskiem, pozostałe trzeba ustawić jako idealnie stałe
(zero wariancji) — `_mad_z` na stałym sygnale zwraca same zera, więc
taki kanał matematycznie nie wnosi nic do E(t). Patrz komentarze w
`demo_scenarios.py`.

## ❌ Odrzucona trzecia wersja (`TIMDRBattery` z "wygaszaniem")

W trakcie budowy tego repo zgłoszono kolejną wersję kodu, próbującą
naprawić powyższe problemy przez dodanie parametru `decay` ("wygaszanie
starych outlierów"). Przetestowano empirycznie przed przyjęciem:

| Test | Wynik |
|---|---|
| Jeden skok w `anomalies()` (n=200, skok na próbce 100) | Oflagował **wszystkie 100 kolejnych próbek** jako anomalie, nie tylko jedną — gorzej niż oryginalny bug |
| `health_score()` po skoku (450 zdrowych próbek po nim) | Nadal **0.000** — `max()` po całej historii wciąż tam jest, "wygaszanie" nic nie zmienia w tej linijce |
| `predict_failure()`, ta sama degradacja + 200 próbek płaskiej historii przed nią | TTF zmienia się z **50.8 na 323.9** — nadal zależy od długości historii, nie tylko od aktualnego stanu |
| `twist()` na n=0/1 | **Crash** (`IndexError`) — zgubiono zabezpieczenie obecne w pierwszej wersji |

Wniosek: centrowanie czasu w tej wersji naprawiło niezależność od
przesunięcia zegara, ale nie naprawiło właściwego problemu (regresja na
całej historii zamiast na oknie), a "wygaszanie" dodane do
`anomalies()`/`health_score()` załatało symptom w złym miejscu, tworząc
nowy błąd zamiast naprawić stary. Repo zostało zbudowane na bazie
pierwszej, w pełni zweryfikowanej poprawki.

## Scenariusze demo

| Scenariusz | Co pokazuje | Domyślny próg |
|---|---|---|
| `normal_cycling` | Zdrowe cykle ładowanie/rozładowanie — rytm | 8.0 |
| `capacity_fade` | Starzenie: rosnąca rezystancja, spadek napięcia — trend + TTF | 20.0 |
| `thermal_event` | Skok temperatury/rezystancji, zapadnięcie napięcia — anomalia + twist | 12.0 |
| `loose_connection` | Powtarzalne skoki rezystancji od wibracji — rytm + twist | 8.0 |
| `internal_short` | Nagły skok prądu, spadek napięcia — anomalia + TTF | 12.0 |

Wynik `python demo.py`:

```
=== normal_cycling ===
  Fusion score: 14.62
  Punkty twist: 23   Anomalie: 0   Okresy rytmu: [20, 40, 60, 80, 100]
  Time-to-failure: 792.2   Health score: 0.878

=== capacity_fade ===
  Fusion score: 18.39
  Punkty twist: 0   Anomalie: 79   Okresy rytmu: []
  Time-to-failure: 6.7   Health score: 0.229

=== thermal_event ===
  Fusion score: 41.40
  Punkty twist: 8   Anomalie: 32   Okresy rytmu: []
  Time-to-failure: 33.1   Health score: 0.498

=== loose_connection ===
  Fusion score: 20.07
  Punkty twist: 38   Anomalie: 19   Okresy rytmu: [20, 40, 60, 80, 100]
  Time-to-failure: brak (inf) - brak trendu ku awarii   Health score: 0.915

=== internal_short ===
  Fusion score: 27.57
  Punkty twist: 8   Anomalie: 64   Okresy rytmu: []
  Time-to-failure: 3.6   Health score: 0.211
```

Uwaga dot. `normal_cycling`: wykryty okres to `20`, nie `40` (prawdziwy
okres cyklu w generatorze) — połowa, z powodu rektyfikacji L2 (fuzja
traci znak). To ten sam, udokumentowany mechanizm co w
TIMDR-Industrial-Fusion, nie błąd tego repo. Zobacz komentarz w
`demo_scenarios.py::normal_cycling`.

## Dashboard + API

```
run.bat
```

albo ręcznie:

```
pip install -r requirements.txt
python api.py
```

potem `http://127.0.0.1:5000`. Dashboard: wybór scenariusza demo,
wczytanie własnego CSV (wymaga dokładnie 4 kolumn: `voltage`, `current`,
`temperature`, `resistance`, plus opcjonalnie kolumna czasu), wykresy
E(t)/trendu/czujników rysowane własnym kodem (Canvas 2D, bez CDN — ten
sam powód co w TIMDR-Industrial-Predict: CDN zablokowany przez
firewall/offline psuje cały dashboard po cichu).

Endpointy: `GET /api/scenarios`, `GET /api/demo?scenario=<id>`,
`POST /api/analyze`, `GET /api/health`.

## Przykład użycia (bez API)

```python
from timdr_battery_fusion import TIMDRBatteryFusion
from timdr_battery_predict import TIMDRBatteryPredict

fusion = TIMDRBatteryFusion()
predict = TIMDRBatteryPredict()

E, Z = fusion.fuse(t, voltage, current, temperature, resistance)
tw_idx, tw_z = fusion.twist(t, E)
tr_sl, tr_z = fusion.trend(t, E, window=30)
an_idx, an_z = fusion.anomalies(E)
periods, r_score = fusion.rhythm(E)

ttf, ttf_lin, ttf_exp = predict.predict_failure(t, E, threshold=8.0, window=60)
health = predict.health_score(E, threshold=8.0, window=20)
```

Uruchomienie testów: `pytest -q` (32 testy: fusion, predict, scenariusze
demo).
