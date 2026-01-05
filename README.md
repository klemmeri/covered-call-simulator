# Covered Call Web Simulator — Backend

This backend provides deterministic strike selection and simple state transitions for a covered-call simulator.

Current scope:
- **Symbol:** SPY only
- **Strike grid:** **$1 increments** (integer strikes)
- **Testing:** via Swagger UI

---

## Run locally (Windows)

```powershell
.\.venv\Scripts\Activate
uvicorn main:app --reload

## API Testing

After starting the server, open:

http://127.0.0.1:8000/docs

Use the Swagger UI to test:
- /covered-call/initialize
- /covered-call/sell-call
- /covered-call/advance-day

Always copy the full response JSON from one step and paste it as the next step’s `state`.
