import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_step_test(df, fname="../plots/step_test_plot.png"):
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    axs[0].step(df.time_hr, df.choke_pct, where="post", color="black")
    axs[0].set_ylabel("Choke (%)")
    axs[0].set_title("Open-Loop Step Test Analysis")
    axs[1].plot(df.time_hr, df.Q_bblhr, color="tab:blue")
    axs[1].set_ylabel("Q (bbl/hr)")
    axs[2].plot(df.time_hr, df.WHP_psi, color="tab:orange")
    axs[2].set_ylabel("WHP (psi)")
    axs[3].plot(df.time_hr, df.FLP_psi, color="tab:green")
    axs[3].set_ylabel("FLP (psi)")
    axs[4].plot(df.time_hr, df.BHP_psi, color="tab:red")
    axs[4].set_ylabel("BHP (psi)")
    axs[4].set_xlabel("Time (hr)")
    for ax in axs:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()


def extract_steady_state(df):
    """
    Identifies steady-state values by taking the last 3 hours of each
    contiguous choke block. FIX: groups by step_id (the actual contiguous
    block), not by choke_pct value, so repeated choke levels in separate
    blocks are never accidentally averaged together.
    """
    df['step_id'] = (df['choke_pct'] != df['choke_pct'].shift()).cumsum()

    tail = df.groupby('step_id').tail(3)
    ss_table = (tail.groupby('step_id')
                .agg(choke_pct=('choke_pct', 'first'),
                     Q_bblhr=('Q_bblhr', 'mean'),
                     WHP_psi=('WHP_psi', 'mean'),
                     FLP_psi=('FLP_psi', 'mean'),
                     BHP_psi=('BHP_psi', 'mean'))
                .sort_values('choke_pct')
                .reset_index(drop=True)
                .round(2))

    # Process gains for ALL active constraint variables, not just Q and WHP
    for col, gain_name in [('Q_bblhr', 'Gain_Q'),
                            ('WHP_psi', 'Gain_WHP'),
                            ('FLP_psi', 'Gain_FLP'),
                            ('BHP_psi', 'Gain_BHP')]:
        ss_table[gain_name] = (ss_table[col].diff() / ss_table['choke_pct'].diff()).round(2)

    print("\n--- DELIVERABLE 1a: STEADY-STATE OPERATING POINTS & GAINS ---")
    print(ss_table.to_string(index=False))
    ss_table.to_csv("../data/steady_state_analysis.csv", index=False)
    print("\nSaved steady_state_analysis.csv")
    return ss_table


def characterize_dynamics(df, tol=0.05):
    """
    For each step change (except the first block, which has no prior state
    to step from), estimates:
      - tau (hr): time to reach 63.2% of the total steady-state change
      - settling time (hr): time after which the response stays within
        `tol` (fractional) of its final steady-state value for the rest
        of the block

    This is what satisfies "dynamic model identification" beyond a plain
    steady-state gain table -- it tells you HOW FAST the well responds,
    which the MPC's prediction horizon needs to account for.
    """
    variables = ['Q_bblhr', 'WHP_psi', 'FLP_psi', 'BHP_psi']
    step_ids = sorted(df['step_id'].unique())
    results = []

    for i, sid in enumerate(step_ids):
        if i == 0:
            continue  # no preceding state to define a step

        block = df[df['step_id'] == sid].reset_index(drop=True)
        prev_block = df[df['step_id'] == step_ids[i - 1]]
        t0 = block['time_hr'].iloc[0]

        row = {'step_id': sid, 'choke_pct': block['choke_pct'].iloc[0],
               'step_start_hr': t0}

        for var in variables:
            initial = prev_block[var].iloc[-1]
            final = block[var].tail(3).mean()
            total_change = final - initial

            if abs(total_change) < 1e-6:
                row[f'{var}_tau_hr'] = 0.0
                row[f'{var}_settle_hr'] = 0.0
                continue

            frac = (block[var] - initial) / total_change

            crossed = block.index[frac >= 0.632]
            tau_hr = (block['time_hr'].iloc[crossed[0]] - t0) if len(crossed) else None
            row[f'{var}_tau_hr'] = tau_hr

            outside = block.index[(frac - 1).abs() > tol]
            if len(outside) == 0:
                settle_hr = 0.0
            else:
                last_outside = outside[-1]
                if last_outside == len(block) - 1:
                    settle_hr = None  # never settled within this block's duration
                else:
                    settle_hr = block['time_hr'].iloc[last_outside + 1] - t0
            row[f'{var}_settle_hr'] = settle_hr

        results.append(row)

    dyn_table = pd.DataFrame(results)
    print("\n--- DELIVERABLE 1b: DYNAMIC RESPONSE (tau = time to 63% of change, "
          f"settle = time within +/-{int(tol*100)}% of final) ---")
    print(dyn_table.to_string(index=False))
    dyn_table.to_csv("../data/dynamic_response_analysis.csv", index=False)
    print("\nSaved dynamic_response_analysis.csv")
    return dyn_table


if __name__ == "__main__":
    raw = pd.read_csv("../data/Autonomous_Choke_Control_Simulated_Dataset.csv")
    df = raw.rename(columns={
        "Time_hr": "time_hr", "Choke_pct": "choke_pct",
        "OilRate_bbl_hr": "Q_bblhr", "WHP_psi": "WHP_psi",
        "FLP_psi": "FLP_psi", "BHP_psi": "BHP_psi",
    })

    plot_step_test(df)
    df['step_id'] = (df['choke_pct'] != df['choke_pct'].shift()).cumsum()
    extract_steady_state(df)
    characterize_dynamics(df)