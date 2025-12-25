# covered-call-simulator

**Interactive covered call simulator for Substack readers (beta)**

This is an educational, interactive simulator designed to explore how covered call strategies behave over time under different market regimes.

It is **not trading advice**.  
All prices are **synthetic**, generated from stochastic processes.  
Nothing here uses real market data or predicts future prices.

The goal is simple:  
to let you *experience* how covered calls behave, one decision at a time.

---

## What this simulator does

- Generates **one synthetic SPY price path per run**
  - No downloads
  - No historical data
  - No peeking into the future
- Reveals prices **one day at a time**
- Lets you decide what to do each day:
  - **[K]eep** the call
  - **[C]lose** the call
  - **[R]oll (fast)** using current defaults
  - **[RR]oll** with new delta / DTE inputs
  - **[Q]uit**
- Supports different market regimes:
  - Choppy
  - Uptrending
  - Downtrending
  - Volatile
  - Mixed
- Shows an interactive plot at each step:
  - Price (left axis)
  - Call strike (red horizontal line)
  - Cash balance (right axis, dashed)
- Tracks running totals:
  - Strategy equity
  - Buy & Hold equity (benchmark)
  - Realized P/L
  - Unrealized P/L
  - Cash
  - Days to expiration

The simulator pauses after each plot and waits for **your decision** before moving on.

---

## What this simulator does *not* do

- It does **not** optimize parameters
- It does **not** run Monte Carlo simulations
- It does **not** cherry-pick paths
- It does **not** know tomorrow’s price
- It does **not** guarantee losses (or gains)

Each run is:
- One random path
- One experience
- One opportunity to see how path-dependence matters

---

## How market regimes work

Market regimes are implemented by changing the **statistical environment** of the random price generator:

- **Drift** (long-term directional bias)
- **Volatility level and clustering**
- **Mean reversion** (for choppy regimes)

The simulator does **not** script outcomes.  
It simply changes the probabilities under which randomness plays out.

Even within the same regime, different runs can produce very different results.

---

## How to run the simulator

### 1) Download the code

Click the green **Code** button → **Download ZIP** → unzip the folder.

You should see the following files in the unzipped folder:

covered_call_simulator.py
requirements.txt
README.md


### 2) Install required Python packages

Open a terminal **inside the unzipped folder that contains `covered_call_simulator.py`**.

On Windows, the easiest way is:
- Open the unzipped folder in File Explorer
- Click the address bar
- Type `cmd` and press Enter

Then run:

pip install -r requirements.txt


### 3) Run the simulator

In the same terminal window, run:

```bash
python covered_call_simulator.py
...

You will be prompted to:

Select a market regime

Enter starting cash

Choose default delta and DTE for new calls

A plot will appear, and the program will pause for your input each day

When the simulation ends, the program creates an **`output` folder in the same directory as the script** and writes one Excel workbook there:  ./output/CC_Simulation_Output.xlsx

Nothing is written anywhere else on your system.

The workbook contains three sheets:
- **EquityCurve** — strategy vs buy-and-hold equity over time
- **TradeLog** — all option trades
- **DecisionLog** — every daily decision you made


## Disclaimer

This simulator is for educational purposes only.  
It is not investment advice.  
Synthetic prices are generated from stochastic processes and do not represent real market behavior.

