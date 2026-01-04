\# Covered Call Web Simulator — Backend



This backend provides deterministic strike selection for a covered-call simulator.

The simulation currently supports \*\*SPY only\*\* with \*\*$1 strike increments\*\*.



---



\## Running the server



```bash

.\\.venv\\Scripts\\Activate

uvicorn main:app --reload

## API Test Flow (Swagger)

1) POST /covered-call/initialize
   - Set spot (e.g., 100)
   - Set start_cash (e.g., 0 or 10000)
   - Execute
   - Copy the full JSON response (this is your state)

2) POST /covered-call/sell-call
   - In request body, replace "state" with the full JSON you copied from initialize
   - Fill "req" with:
     - strike (e.g., 105)
     - dte_days (e.g., 1)
     - delta_target (e.g., 0.3)
     - credit_per_share (e.g., 1.25)
   - Execute
   - Confirm response shows short_call populated and cash increased

3) POST /covered-call/advance-day
   - In request body, replace "state" with the full JSON you copied from sell-call response
   - Set req fields (spot/price as required by your schema)
   - Execute
   - Confirm assignment/expiration behavior matches expectations

Tip: Always copy the entire response JSON from the prior step and paste it as the next step’s "state".

# Covered Call Web Simulator — Backend

Deterministic strike selection endpoints for a covered-call simulator.

## Assumptions (current)
- Symbol is fixed to **SPY** (only ETF supported in this simulation).
- Strike grid is **$1 increments** (integer strikes).
- User selects the **starting spot price**.

## Run locally
```bash
.\.venv\Scripts\Activate
uvicorn main:app --reload
