# SentinelFlow MPC: Autonomous Production Choke Controller

> **Project:** Autonomous Production Choke Controller for a Single Naturally Flowing Oil Well

**⚠️ Note on Hackathon Submission:** 
**Due to the submission portal's strict security firewall blocking the upload of executable source code (including `.py` and `.txt` formats), only the project presentation could be uploaded to the portal before the deadline.** 

**This repository contains the complete, timestamped source code for the SentinelFlow MPC system, the Mock Simulator, and the generated datasets. All code in this repository was finalized and pushed prior to the official hackathon deadline.**


##  Overview

**SentinelFlow MPC** is a physics-aware Model Predictive Control (MPC) architecture designed to autonomously regulate production choke valves in naturally flowing oil and gas wells. By combining mathematical system identification (FOPDT) with machine learning surrogate models, this controller dynamically tracks target flow rates while strictly enforcing physical safety boundaries to prevent liquid loading and asset degradation.

Unlike conventional black-box AI models that can produce physically impossible predictions, SentinelFlow MPC incorporates a **Hardware Integrity Filter** that constrains every control decision within validated physical operating limits before execution.


##  Core Features & Technical Innovations

- **Hybrid Multi-Step MPC Horizon**
  - Executes every **1 hour**
  - Predicts well behavior **4 hours ahead**
  - Anticipates delayed pressure responses before safety violations occur.

- **Physics-Aware Safety Enforcement**
  - Validates future predictions of:
    - Wellhead Pressure (WHP)
    - Flowline Pressure (FLP)
    - Bottom Hole Pressure (BHP)
  - Rejects unsafe control actions before implementation.

- **Hardware Safety Clip**
  - Uses

  ```python
  MAX_TRAINED_CHOKE = 65.0
  ```

  to prevent dangerous out-of-distribution AI extrapolation.

- **Target Flow Tracking**
  - Automatically adjusts choke position to reach requested production while respecting all operational constraints.

- **Edge Deployment Ready**
  - Lightweight mathematical models
  - No GPU required
  - Runs locally on wellsite gateway devices
  - Operates without continuous internet connectivity


# System Architecture

```
              Real-Time Well Parameters
 ┌─────────────────────────────────────────────┐
 │ Oil Flow Rate (Q)                           │
 │ Wellhead Pressure (WHP)                     │
 │ Flowline Pressure (FLP)                     │
 │ Bottom Hole Pressure (BHP)                  │
 │ Current Choke Position                      │
 └─────────────────────────────────────────────┘
                     │
                     ▼
        Machine Learning Surrogate Model
                     │
                     ▼
        Hardware Integrity Filter
      (Physics-Aware Safety Validation)
                     │
                     ▼
      Hybrid Multi-Step MPC Optimizer
          (4-Hour Prediction Horizon)
                     │
                     ▼
      Next Safe Choke Position Command
```


#  Autonomous Control Workflow

The controller executes a closed-loop optimization cycle every hour.

## Step 1 — System Identification

The system performs step tests to estimate process dynamics using First-Order Plus Dead Time (FOPDT) modeling.

Outputs include:

- Process Gain (K)
- Time Constant (τ)
- Dead Time

These parameters characterize well behavior.


## Step 2 — Multi-Step Prediction

Using the identified model, multiple candidate choke moves are generated.

Each candidate is simulated across a **4-hour prediction horizon**.

Future values predicted include:

- Oil Flow Rate
- WHP
- FLP
- BHP


## Step 3 — Physics Safety Enforcement

Every simulated future state is validated against physical safety limits.

If **any** predicted state violates minimum constraints, the candidate is immediately rejected.

Safety constraints include:

- Minimum WHP
- Minimum FLP
- Minimum BHP


## Step 4 — Execution

The safest feasible choke percentage is applied.

The controller waits one hour for the well response before repeating the optimization cycle.


#  Predictive Safety Enforcement

The controller evaluates future well states before approving a choke movement.

```python
# CHOKE_CONTROLLER.PY: Predictive Safety Enforcement
def evaluate_candidate_move(self, candidate_choke):
    # Lookahead: 4-hour MPC horizon to catch delayed pressure crashes
    future_states = self.predict_rollout(candidate_choke, steps=4)
    
    for state in future_states:
        # Veto move if any future step violates physical hardware limits
        if state.BHP < MIN_BHP or state.WHP < MIN_WHP or state.FLP < MIN_FLP:
            self.log_warning(f"Vetoed {candidate_choke}%: Safety limit breached.")
            return False 
            
    return True # Move structurally safe, approved for execution
```


#  Hardware Safety Clip

To eliminate AI extrapolation beyond trained operating conditions:

```python
MAX_TRAINED_CHOKE = 65.0

candidate = min(candidate, MAX_TRAINED_CHOKE)
```

This guarantees the controller never recommends choke openings beyond validated operating ranges.

---

#  Performance Scenarios

## Scenario A — Startup to Target

- Gradually increases production
- Smoothly reaches target flow
- Maintains safe pressure margins
- No oscillations

<img width="1000" height="1000" alt="scenario_a" src="https://github.com/user-attachments/assets/ed163073-f201-4549-a278-6554788716a7" />


```
plots/scenario_a.png
```


## Scenario B — Active Safety Override

Requested production exceeds safe operating limits.

Instead of blindly maximizing production, SentinelFlow MPC intentionally limits flow to preserve well integrity.

Result:

- Lower production
- Safe operation
- No structural violations

<img width="1000" height="1000" alt="scenario_b" src="https://github.com/user-attachments/assets/70bcf4a6-af35-41e0-a1cd-11f9418f846e" />


```
plots/scenario_b.png
```


## Scenario C — Hard-Clip Veto

The requested choke exceeds trained operating limits.

The controller activates:

```
MAX_TRAINED_CHOKE = 65%
```

Unsafe commands are rejected before execution.

Result:

- No AI hallucination
- No unsafe extrapolation
- Physically valid operation

<img width="1000" height="1000" alt="scenario_c" src="https://github.com/user-attachments/assets/4d4f3b1d-9fcd-4a04-8f47-d64d3cf68653" />


```
plots/scenario_c.png
```


# System Outputs

The controller continuously monitors:

- Oil Flow Rate (Q)
- Wellhead Pressure (WHP)
- Flowline Pressure (FLP)
- Bottom Hole Pressure (BHP)
- Choke Position

Typical dashboard includes:

- Startup Response
- Target Tracking
- Safety Override
- Infeasible Target Rejection


# Repository Structure

```text

SentinelFlow-MPC
│
├── data/
│   ├── Autonomous_Choke_Control_Simulated_Data.xlsx  # Baseline simulated well parameters
│   ├── full_range_step_test_data.xlsx                # Pressure/flow reactions across 0-100% choke range
│   └── steady_state_analysis.xlsx                    # Stabilized well conditions post-step changes
│
├── plots/
│   ├── scenario_a.png                                # Graph of normal operation and target tracking
│   ├── scenario_b.png                                # Graph of active safety throttling
│   ├── scenario_c.png                                # Graph of infeasible target rejection
│   └── step_test_plot.png                            # Visual mapping of dynamic pressure inertia
│
├── src/
│   ├── generate_full_range_data.py                   # Script to generate physical well boundary datasets
│   ├── step_test.py                                  # Executes open-loop dynamic FOPDT mapping
│   ├── mock_simulator.py                             # Simulates well thermodynamics and flow responses
│   ├── controller.py                                 # Core MPC multi-step lookahead and veto logic
│   └── surrogate_model.py                            # Scikit-Learn linear models for predictive states
│
├── README.md                                         # Project documentation
└── requirements.txt                                  # Python dependencies
```


#  Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-Learn |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Control Algorithm | Custom Multi-Step MPC |
| Dynamic Modeling | FOPDT |


#  Installation

Clone the repository

```bash
git clone https://github.com/YourUsername/SentinelFlow-MPC.git
```

Move into the project

```bash
cd SentinelFlow-MPC
```

Install dependencies

```bash
pip install -r requirements.txt
```


# Usage

Generate system identification data

```bash
python src/generate_full_range_data.py
```

Run step-test analysis

```bash
python src/step_test.py
```

Execute the MPC controller

```bash
python src/controller.py
```


# Industrial & Academic Validation

### Optimizing Gas Wells

Supports automated choke optimization and real-time production management.

---

### Robust Automatic Well Choke Control

Demonstrates constraint-based automatic choke regulation while preserving operational safety.


### First-Order Plus Dead Time (FOPDT)

Provides the mathematical foundation for:

- Process Gain estimation
- Dead Time estimation
- Pressure dynamics
- Time Constant identification


# Key Contributions

- Physics-aware AI controller
- Hardware Integrity Filter
- Multi-Step Predictive Control
- Out-of-Distribution Safety Clip
- Edge Deployable Architecture
- Autonomous Production Optimization
- Industrial Safety Enforcement


# License

This project was developed as part of the Honeywell Hackathon software innovation challenge.

It is intended for academic, research, and educational purposes to demonstrate advanced industrial control logic and machine learning applications in the oil and gas sector.
