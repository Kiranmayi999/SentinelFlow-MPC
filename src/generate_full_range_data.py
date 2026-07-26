import numpy as np
import pandas as pd
from mock_simulator import MockSimulator


def generate_step_test_data(choke_start=0, choke_end=100, choke_step=5,
                             hours_per_step=20, seed=42):
    """
    Runs the mock simulator across a range of fixed choke levels,
    holding each level long enough to reach (its version of) steady state.
    """
    np.random.seed(seed)
    sim = MockSimulator()

    records = []
    t = 0
    choke_levels = np.arange(choke_start, choke_end + choke_step, choke_step)

    for choke in choke_levels:
        for _ in range(hours_per_step):
            q, whp, flp, bhp = sim.step(float(choke))
            records.append({
                'Time_hr': t,
                'Choke_pct': choke,
                'OilRate_bbl_hr': round(q, 2),
                'WHP_psi': round(whp, 2),
                'FLP_psi': round(flp, 2),
                'BHP_psi': round(bhp, 2),
                'In_Trained_Range': 30 <= choke <= 65  # flag for downstream analysis
            })
            t += 1

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_step_test_data(
        choke_start=0,
        choke_end=100,
        choke_step=5,
        hours_per_step=20,
    )

    df.to_csv("../data/full_range_step_test_data.csv", index=False)

    print(f"Saved {len(df)} rows -> full_range_step_test_data.csv")
    print(f"Choke levels tested: {sorted(df.Choke_pct.unique())}")
    n_out_of_range = (~df['In_Trained_Range']).sum()
    print(f"\nRows OUTSIDE the model's trained 30-65% range: {n_out_of_range} "
          f"({100*n_out_of_range/len(df):.0f}% of data)")
    print("Treat those rows as model extrapolation, not validated physics.")