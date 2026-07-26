import matplotlib.pyplot as plt
import os
import numpy as np
from mock_simulator import MockSimulator
from choke_controller import AutonomousChokeController

def run_scenario(sim, controller, initial_target_q, scenario_name, filename):
    print(f"\n--- Running {scenario_name} ---")
    hours = 50
    history = {'t':[], 'target_q':[], 'q':[], 'whp':[], 'flp':[], 'bhp':[], 'choke':[]}
    
    current_choke = sim.current_choke
    target_q = initial_target_q
    
    for t in range(hours):
        # Scenario B logic: Change target flow rate exactly halfway through
        if scenario_name == "Scenario B: Target Tracking" and t == 25:
            target_q = 145.0
            
        # 1. Controller decides the next move
        next_choke = controller.compute_next_move(
            current_choke, target_q, 
            sim.current_q, sim.current_whp, sim.current_flp, sim.current_bhp
        )
        
        # 2. Simulator executes the move
        q, whp, flp, bhp = sim.step(next_choke)
        current_choke = next_choke
        
        # 3. Log the results
        history['t'].append(t)
        history['target_q'].append(target_q)
        history['q'].append(q)
        history['whp'].append(whp)
        history['flp'].append(flp)
        history['bhp'].append(bhp)
        history['choke'].append(current_choke)

   # 4. Generate the Plot (Now 4 rows instead of 3)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # TOP PLOT: Flow Rates
    axs[0].plot(history['t'], history['target_q'], 'r--', label='Target Oil Rate')
    axs[0].plot(history['t'], history['q'], 'b-', label='Actual Oil Rate (Q)')
    axs[0].set_ylabel('Flow (bbl/hr)')
    axs[0].legend(loc='lower right')
    axs[0].set_title(scenario_name)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    # SECOND PLOT: Surface Pressures (WHP & FLP)
    axs[1].plot(history['t'], history['whp'], 'g-', label='WHP (Wellhead)')
    axs[1].plot(history['t'], history['flp'], 'y-', label='FLP (Flowline)')
    axs[1].axhline(y=controller.MIN_WHP, color='r', linestyle=':', label='Min Safe WHP')
    axs[1].axhline(y=controller.MIN_FLP, color='darkorange', linestyle=':', label='Min Safe FLP')
    axs[1].set_ylabel('Surface (psi)')
    # Use fontsize='small' to ensure all 4 labels fit nicely in the legend
    axs[1].legend(loc='lower right', fontsize='small')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    # THIRD PLOT: Bottom Hole Pressure (BHP)
    axs[2].plot(history['t'], history['bhp'], 'm-', label='BHP (Bottom Hole)')
    axs[2].axhline(y=controller.MIN_BHP, color='r', linestyle=':', label='Min Safe BHP')
    axs[2].set_ylabel('Bottom (psi)')
    axs[2].legend(loc='lower right')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    # BOTTOM PLOT: Choke Position
    axs[3].plot(history['t'], history['choke'], 'k-', drawstyle='steps-post', label='Choke Position (%)')
    axs[3].set_ylabel('Opening (%)')
    axs[3].set_xlabel('Time (Hours)')
    axs[3].legend(loc='lower right')
    axs[3].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Save the plot to the plots folder
    os.makedirs('../plots', exist_ok=True)
    plt.savefig(f'../plots/{filename}')
    print(f"Saved plot: ../plots/{filename}")

if __name__ == "__main__":
    # np.random.seed(42)
    # The MockSimulator will now automatically find the CSV file on its own!
    
    # Scenario A: Startup to Target (Safe target of 120 bbl/hr)
    sim_a = MockSimulator()
    ctrl_a = AutonomousChokeController(sim_a)
    run_scenario(sim_a, ctrl_a, 120.0, "Scenario A: Startup to Target", "scenario_a.png")
    
    # Scenario B: Target Tracking (Starts at 100, jumps to 145 mid-run)
    sim_b = MockSimulator()
    ctrl_b = AutonomousChokeController(sim_b)
    run_scenario(sim_b, ctrl_b, 100.0, "Scenario B: Target Tracking", "scenario_b.png")
    
    # Scenario C: Infeasible Target (Asks for 200 bbl/hr, which is unsafe)
    sim_c = MockSimulator()
    ctrl_c = AutonomousChokeController(sim_c)
    run_scenario(sim_c, ctrl_c, 200.0, "Scenario C: Infeasible Target", "scenario_c.png")