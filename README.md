# GeoAI Outdoor Activity Recommendation Assistant
**CSRE, IIT Bombay — GeoAI Hackathon Project**

An evidence-grounded outdoor activity recommendation assistant powered by live geospatial and meteorological data coupled with an open-weights Small Language Model (SLM &le; 3B parameters).

---

## 1. Project Overview & Problem Statement

Standard LLMs hallucinate current local conditions. When asked:
- *"Where should I go running tomorrow morning?"*
- *"Which park is best for a picnic this weekend?"*
- *"Is it safe for me to cycle to work at 8 AM?"*
- *"When should children play cricket outdoors?"*
- *"Where can my parents take a comfortable walk tomorrow?"*

They cannot guarantee whether ground surfaces are wet, if air pollution (PM2.5) is dangerous for someone with asthma, or whether adequate daylight and tree canopy shade exist.

This assistant answers these questions by:
1. Parsing user intent, target activity, and personal vulnerabilities (children, seniors, asthma, mobility, pregnancy).
2. Geocoding the location in real-time using **Nominatim / OpenStreetMap**.
3. Discovering candidate parks, pitches, paths, viewpoints, and recreational areas using the **Overpass OSM API**.
4. Fetching real-time weather forecasts (temperature, apparent heat, rain probability, wind vectors) from **Open-Meteo Forecast API**.
5. Fetching particulate air quality (PM2.5, PM10, European & US AQI) from **Open-Meteo Air Quality API**.
6. Calculating street distances and estimated travel times using **OSRM (Open Source Routing Machine)**.
7. Calculating sunrise, sunset, and photographic Golden Hour windows.
8. Evaluating and ranking candidates through **10 activity-specific suitability engines** with hard safety constraints.
9. Generating natural language recommendations using an open-weights SLM (**Qwen2.5-1.5B-Instruct**, 1.54B parameters) strictly grounded on programmatic evidence.

---

## 2. 10 Supported Outdoor Activities (Rulebook Section 3)

| Activity | Key Meteorological & Geospatial Considerations | Disqualification Gates |
| :--- | :--- | :--- |
| **1. Running / Jogging** | Optimal temp 12–22°C, PM2.5 breathing stress, paved surface | Apparent temp >36°C, PM2.5 >40 (Asthma >25) |
| **2. Cycling** | Wind direction & speed (<24 km/h), wet road skid risk, visibility | Rain >1.0 mm/h, Wind >30 km/h |
| **3. Golf** | Wind ball drift (<22 km/h), daylight horizon, course availability | Rain >0.5 mm/h, Sunset within 2 hours |
| **4. Cricket** | Pitch waterlogging, heat stress in batting pads & helmet | Rain >0.2 mm/h, wet pitch hazard |
| **5. Football** | Turf surface condition, high cardio thermal limits | Heavy rain >3.5 mm/h, Apparent temp >38°C |
| **6. Tennis** | Ball waterlogging, court surface friction & lateral slip | Rain >0.1 mm/h, Wind >20 km/h |
| **7. Hiking / Walk** | Daylight cutoff, terrain grade, tree canopy shade | Thunderstorm code &ge;95, Sunset cutoff |
| **8. Picnic** | Zero rain tolerance, wind <18 km/h (napkins/food), benches | Rain >0.0 mm/h, High wind |
| **9. Outdoor Events** | Weather across multi-hour event span, crowd thermal safety | Rain >0.5 mm/h, Lightning |
| **10. Photography** | Golden hour (+20 bonus), 20–60% cloud cover, low wind | 100% thick rain overcast |

---

## 3. Small Language Model (SLM) Specification (Rulebook Section 7)

- **Selected Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Parameter Count:** 1.54 Billion (Fully compliant with &le; 3B parameter limit)
- **License:** Apache 2.0 (Open Weights)
- **Context Length:** 32,768 tokens
- **Hugging Face Hub Repository:** [https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Alternative Eligible Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct` (1.71B parameters)

### Loading Workflow (`from_pretrained`)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

---

## 4. Live Data Sources & Attribution

1. **Geocoding:** Nominatim / OpenStreetMap (`© OpenStreetMap contributors`, ODbL)
2. **Venues & POIs:** Overpass API (`© OpenStreetMap contributors`)
3. **Weather Forecast:** Open-Meteo Forecast API (`Open-Meteo.com`, non-commercial open data)
4. **Air Quality:** Open-Meteo Air Quality API (Copernicus Atmosphere Monitoring Service)
5. **Routing & Distances:** Project OSRM / OpenStreetMap

---

## 5. Quickstart & Installation

### Requirements
- Node.js &ge; 18
- npm / bun

### Setup
```bash
# 1. Clone repository
git clone <repo-url>
cd geoai-outdoor-assistant

# 2. Install dependencies
npm install

# 3. Run unit tests (suitability rules & safety limits)
npm test

# 4. Start development server
npm run dev
# The application runs on http://localhost:3000
```

---

## 6. HTTP API Documentation for Hackathon Judges

### 1. `POST /api/recommend`
Generates full geospatial outdoor recommendation pipeline.

**Request:**
```bash
curl -X POST http://localhost:3000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where should I go running tomorrow morning?",
    "locationQuery": "Powai, Mumbai",
    "activity": "running",
    "constraints": {
      "hasAsthma": true,
      "preferredRadiusKm": 5
    }
  }'
```

**Response Format:**
```json
{
  "query": "Where should I go running tomorrow morning?",
  "activity": "running",
  "resolvedLocation": {
    "name": "Powai",
    "lat": 19.1334,
    "lon": 72.9133
  },
  "bestRecommendation": {
    "place": {
      "name": "Powai Lake Promenade & Garden",
      "distanceKm": 1.8,
      "travelTimeMinutes": 8
    },
    "recommendedTimeWindow": "07:00 - 09:00",
    "suitabilityScore": 92,
    "whyRecommended": [...]
  },
  "alternatives": [...],
  "liveConditions": {
    "weather": { "temperatureC": 22.4, "windSpeedKmh": 11.2, "precipitationMm": 0 },
    "airQuality": { "pm2_5": 21.0, "category": "Moderate" },
    "daylight": { "sunrise": "06:15", "sunset": "18:25" }
  },
  "preparation": { "clothing": [...], "equipment": [...], "hydrationAndNutrition": [...] },
  "risksAndAvoidConditions": { "whenToAvoid": [...], "vulnerabilityAlerts": [...] },
  "slmReasoning": { "modelName": "Qwen2.5-1.5B-Instruct", "generatedText": "..." }
}
```

### 2. `GET /api/health`
```bash
curl -s http://localhost:3000/api/health
```

### 3. `GET /api/model-info`
```bash
curl -s http://localhost:3000/api/model-info
```

---

## 7. Testing & Evaluation

Run the automated test suite verifying all 10 activities, edge cases, and safety disqualifications:
```bash
npm test
```
Tests assert:
- Activity profile validity for all 10 sports & activities.
- Rain and wet-surface unviability for cricket and tennis.
- Golden hour lighting bonus for photography (+20 score boost).
- Asthma safety constraint disqualification on elevated PM2.5 (>50 µg/m³).
- Extreme heat (>38°C feels-like) heatstroke safety disqualification.
- Mobility wheelchair accessibility bonuses and rough terrain warnings.
