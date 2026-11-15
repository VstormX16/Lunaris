# Lunaris

**A lunar landing site analyzer for the Moon's south pole.** Pick a date, a spacecraft and a place on the Moon, and Lunaris tells you whether the mission would work: how much sunlight there is, whether Earth can be seen for radio contact, how steep the ground is, whether the battery gets through the night, and, in the end, a plain verdict with the chance of success.

Built for the NASA Space Apps Challenge (team TetraTech). Every number comes from real NASA data. There is no synthetic science data, and all machine learning, including the assistant's language model, is written from scratch in NumPy.

> Türkçe kısa özet için en alta bakın: [Türkçe özet](#türkçe-özet).

---

## What it does

| Page | What you get |
|---|---|
| **New Operation** | A guided analysis of one scenario (date, vehicle, place), or a comparison of two. The result page gives a **verdict** (Go / Possible with risks / Not recommended / Do not proceed), a 0–100 **score**, the **mission success probability**, the reasons, and charts: the month of Sun and Earth visibility, slope against the vehicle's limit, a shaded terrain picture, the sky from the site, and **Earth's wobble (libration)**. A PDF report can be downloaded. |
| **Mission Planner** | The full analysis for *every* possible start date in the next 60–180 days, painted on a calendar, with the best launch windows. Open a date and the analysis page shows exactly what the calendar showed. |
| **South Area** | A map of the whole south-polar cap scored by our neural network, with an in-page check of a chosen spot for a chosen vehicle. |
| **Moon Map**, **Space Time**, **Moon Timeline**, **360 View** | Pick any point on the Moon: Sun and Earth positions, sunrise and sunset, a 3D Moon you can move through time, and a first-person view from 2 m above the ground with real cast shadows. |
| **Spacecrafts** | A garage of 52 lunar missions (1959–2026) with specifications, photos and which values are verified. You can also **design your own vehicle** (battery, solar array, loads, slope limit, heater, communications) and see at once how it would do; it then works in every other page. |
| **AI Assistant** | Ask in Turkish or English, in your own words: "On 16 November 2026 I want to go to the south pole, which vehicle suits best?", "What should a lander have to survive at Shackleton?". It also answers questions about the open analysis. |
| **History** | Every analysis and conversation is saved locally and can be reopened. |

## Where the numbers come from

All inputs are real NASA data; nothing is simulated or made up.

- **Positions of the Sun and Earth:** NASA NAIF SPICE, the DE440 ephemeris, in the Moon's own frame with its real orientation (`moon_pa_de440`, which includes the physical **libration**), light-time and aberration corrected.
- **Terrain:** NASA LRO LOLA. A 40 m grid south of 85°S and the 470 m whole-Moon LDEM_64 grid everywhere else. The horizon, shadows, slopes and Earth line-of-sight are computed from it by ray tracing.
- **Cross-check:** NASA JPL Horizons.
- **Vehicles:** a dataset of the lunar landers and rovers. Each value is marked *verified* (checked against an agency source), *unverified* (from the team dataset) or *unknown*. Unknown values are never guessed; that check is reported as "not assessed" and lowers the confidence.

### The verdict and the probability

The **score** rates the whole 30-day window with the vehicle's own limits (slope, power, cold, communications), and hard limits cap it. The **success probability** re-runs the mission 400 times under the uncertainty that really exists: where in the landing area it touches down (real LOLA slopes), a landing up to 3 days late (real SPICE Sun/Earth), the vehicle's figures varied (±5 % when verified, ±20 % when not), and how long unheated electronics survive the cold (48–120 h). The battery is simulated hour by hour; an empty battery means hibernation, not the end, unless the outage outlasts the cold limit. A run succeeds if the lander touches down within its slope limit, works at least 7 days in daylight and can send its data home. These spreads are assumptions of this tool, not vehicle data, and are echoed in the results. The simulation was back-tested on real landings (Chandrayaan-3, Luna 9, Chang'e 4); see `tests/test_mission_risk.py`.

### Machine learning from scratch

No pre-trained models, no hosted AI services, no PyTorch, TensorFlow or scikit-learn.

- **Landing-site surrogate network** (`computation_core/scratch_nn`): an ensemble of networks written in NumPy (layers, Adam, AutoML with successive halving), trained on labels computed by the exact physics engine. It draws the South Area map and decides which pixels get the exact physics.
- **The assistant** (`computation_core/assistant`): a small intent network on hashed word and character n-grams (typo and Turkish-diacritic tolerant, trained in NumPy), hand-written entity rules for dates, places and vehicles, a concept parser for whole mission plans, and a dialogue manager that calls the real engine. Every number it says was just computed.

---

## Requirements

- **Python 3.10 or newer** (developed on 3.10)
- **Node.js 20 or newer** (developed on 24) and npm
- About **1 GB** of free disk space and an internet connection for the first run (about 650 MB of NASA data is downloaded once)
- Windows, macOS or Linux (the one-click launcher `start.bat` is Windows only)

## Installation

```bash
# 1. get the code
git clone https://github.com/VstormX16/Lunaris.git
cd Lunaris

# 2. Python environment and dependencies
python -m venv .venv
#   Windows:      .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 3. download the NASA data (once, about 650 MB: SPICE kernels, LOLA 40 m polar grid, LOLA whole-Moon grid)
python -m data_engine

# 4. the web app's dependencies
cd frontend
npm install
cd ..
```

The trained models (the landing-site network and the assistant's language model, about 40 MB) are included in the repository, so nothing needs to be trained. See [Retraining](#retraining-optional) if you want to.

`.env` is optional. A NASA API key is not needed for anything the app does today; if you want one anyway, copy `.env.example` to `.env`.

## Running

**Windows:** double-click `start.bat`. It opens two windows (the analysis server and the web app); open <http://localhost:5173> when both say they are ready.

**Any system**, in two terminals from the project folder:

```bash
# terminal 1: the analysis server (port 8000)
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000

# terminal 2: the web app (port 5173)
cd frontend
npm run dev
```

Then open <http://localhost:5173>. The server needs a few seconds at start-up to load the kernels and the elevation grids. Interactive API documentation is at <http://127.0.0.1:8000/docs>. To point the web app at another server, set `VITE_API_URL` before `npm run dev`.

A production build of the web app is made with `cd frontend && npm run build`.

## Tests

```bash
python -m pytest tests -q
```

About 160 tests (a few minutes). They run against the real NASA data, so run `python -m data_engine` first; without the data they are skipped. They cover the physics (ray-traced horizons, Sun and Earth geometry, libration), the neural network, the assistant (including sentences it has never seen), the mission simulation with its back-test on real landings, the API, and the planner.

## Retraining (optional)

```bash
# the landing-site network: builds the training data from the exact physics, then searches and trains (about 30-40 min)
python -m computation_core.dataset_builder --split train
python -m computation_core.dataset_builder --split test
python -m computation_core.scratch_nn.automl            # add --quick for a short smoke run

# the assistant's language model (about 6 min)
python -m computation_core.assistant.train_nlu
```

If you edit the spacecraft source file ("Uzay Araç Güncel Bilgileri.txt"), rebuild the garage with `python -m data_engine.build_spacecrafts`.

## Project layout

```
api/                 FastAPI server (analysis, planner, vehicles, libration, assistant, history)
computation_core/    The physics and the models
  horizon.py, site_analysis.py, ...   ray-traced horizon, Sun/Earth/slope metrics from SPICE + LOLA
  vehicle_assessment.py, mission_risk.py, verdict.py, planner.py, libration.py
  scratch_nn/        the neural network library and the landing-site model (NumPy)
  assistant/         the AI assistant (NumPy intent model, entity rules, dialogue)
data_engine/         NASA downloads (SPICE, LOLA, Horizons, Trek) and the spacecraft dataset builders
frontend/            React + TypeScript + Vite + Tailwind web app (Three.js, Leaflet, jsPDF)
tests/               pytest suites
data/                downloaded NASA data and local databases (not in the repository, except the trained models)
```

## Good to know

- Your History is stored on your own machine in `data/operations.sqlite3` and never leaves it. Vehicles you design are stored there too.
- The 40 m terrain exists south of 85°S. Elsewhere on the Moon the analysis works on the 470 m grid, so slopes are regional and small hills are smoothed away; the app says which one a result used. The neural South Area map covers the polar cap only.
- Most vehicle figures in the garage are from the team's dataset and are marked unverified. The simulation treats them that way, and the confidence says so.
- The spacecraft photos were collected from public sources; check each licence before redistributing them.

## Credits and data sources

- NASA NAIF SPICE kernels (DE440, lunar orientation), JPL Horizons, LRO LOLA elevation data (PDS Geosciences Node), NASA Moon Trek tiles, NASA Scientific Visualization Studio Earth imagery.
- The Geom typeface (SIL Open Font License; see `Geom/OFL.txt`).

---

## Türkçe özet

**Lunaris**, Ay'ın güney kutbunda iniş yeri analizi yapar: tarih, uzay aracı ve konum seçersin; Güneş ve Dünya'nın görünürlüğü, eğim, gece ve pil hesaplanır ve sonunda net bir karar ile görevin başarı şansı verilir. Tüm sayılar gerçek NASA verisinden (SPICE, LOLA) gelir; yapay zekâ (iniş yeri ağı ve Türkçe/İngilizce asistan) sıfırdan NumPy ile yazılmıştır.

**Kurulum (özet):**

```bash
git clone https://github.com/VstormX16/Lunaris.git
cd Lunaris
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m data_engine                                # NASA verisi, bir kez, yaklaşık 650 MB
cd frontend && npm install && cd ..
```

**Çalıştırma:** Windows'ta `start.bat`'a çift tıkla. Diğer sistemlerde iki terminalde `python -m uvicorn api.server:app --host 127.0.0.1 --port 8000` ve `cd frontend && npm run dev` komutlarını çalıştırıp <http://localhost:5173> adresini aç.

**Testler:** `python -m pytest tests -q` (önce NASA verisi indirilmiş olmalı).
