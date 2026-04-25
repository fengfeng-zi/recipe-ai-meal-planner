# Recipe Studio Web

Web demo for the full meal workflow:

`upload image -> analyze dish -> retrieve recipe evidence -> generate meal plan`

## Run

From repo root:

```powershell
python recipe_studio_web\server.py
```

Module entrypoint also works:

```powershell
python -m recipe_studio_web
```

Default local URL:

```text
http://127.0.0.1:8787
```

## Verification

```powershell
python recipe_studio_web\tests.py
```

What it checks:

- backend endpoints return valid JSON payloads
- multipart upload returns a saved image path and image analysis
- retrieval response includes evidence-card fields
- plan response includes structured `plan_days`
- page exposes preview/progress/plan-card containers
