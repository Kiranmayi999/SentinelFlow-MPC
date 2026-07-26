import numpy as np
import pandas as pd

class AutonomousChokeController:
    def __init__(self, simulator_model):
        self.sim = simulator_model
        
        # 1. HARD CONSTRAINTS
        self.MAX_RAMP_RATE = 5.0  
        self.MAX_TRAINED_CHOKE = 65.0  
        
        # 2. SAFETY OPERATING ENVELOPE
        self.MIN_WHP = 225.0      
        self.MIN_FLP = 160.0      
        self.MIN_BHP = 2950.0     
        
        # 3. SAFETY MARGINS
        self.SAFETY_MARGIN_WHP = 5.0
        self.SAFETY_MARGIN_FLP = 6.0
        self.SAFETY_MARGIN_BHP = 15.0

    def _predict_candidate(self, candidate_choke, curr_q, curr_whp, curr_flp, curr_bhp):
        """Predicts exactly 1 hour into the future."""
        X_input = pd.DataFrame([[
            candidate_choke, curr_q, curr_whp, curr_flp, curr_bhp
        ]], columns=['Choke_pct', 'Prev_Q', 'Prev_WHP', 'Prev_FLP', 'Prev_BHP'])
        
        pred_q = self.sim.model_Q.predict(X_input)[0]
        pred_whp = self.sim.model_WHP.predict(X_input)[0]
        pred_flp = self.sim.model_FLP.predict(X_input)[0]
        pred_bhp = self.sim.model_BHP.predict(X_input)[0]
        
        return pred_q, pred_whp, pred_flp, pred_bhp

    def _predict_horizon(self, candidate_choke, curr_q, curr_whp, curr_flp, curr_bhp, steps=4):
        """Rolls the model forward N steps to catch delayed momentum crashes."""
        q, whp, flp, bhp = curr_q, curr_whp, curr_flp, curr_bhp
        trajectory = []
        for _ in range(steps):
            q, whp, flp, bhp = self._predict_candidate(candidate_choke, q, whp, flp, bhp)
            trajectory.append((q, whp, flp, bhp))
        return trajectory

    def compute_next_move(self, current_choke, target_q, curr_q, curr_whp, curr_flp, curr_bhp):
        # 1. EMERGENCY OVERRIDE
        if (curr_whp < self.MIN_WHP + self.SAFETY_MARGIN_WHP or 
            curr_flp < self.MIN_FLP + self.SAFETY_MARGIN_FLP or 
            curr_bhp < self.MIN_BHP + self.SAFETY_MARGIN_BHP):
            return max(0.0, current_choke - self.MAX_RAMP_RATE)
            
        candidates = [
            current_choke - self.MAX_RAMP_RATE,
            current_choke - (self.MAX_RAMP_RATE / 2.0),
            current_choke,
            current_choke + (self.MAX_RAMP_RATE / 2.0),
            current_choke + self.MAX_RAMP_RATE
        ]
        
        best_choke = None
        smallest_error = float('inf')
        
        for move in candidates:
            # Clamp to physical and model-trained limits
            move = np.clip(move, 0.0, self.MAX_TRAINED_CHOKE)
            
            # Look 4 hours into the future
            trajectory = self._predict_horizon(move, curr_q, curr_whp, curr_flp, curr_bhp, steps=4)
            
            # SAFETY CHECK: Reject if ANY future step in the horizon breaches the margin
            is_safe = True
            for fut_q, fut_w, fut_f, fut_b in trajectory:
                if (fut_w < self.MIN_WHP + self.SAFETY_MARGIN_WHP or 
                    fut_f < self.MIN_FLP + self.SAFETY_MARGIN_FLP or 
                    fut_b < self.MIN_BHP + self.SAFETY_MARGIN_BHP):
                    is_safe = False
                    break 
            
            if not is_safe:
                continue
                
            # If the entire 4-hour horizon is safe, evaluate accuracy based on the next immediate step
            pred_q = trajectory[0][0] 
            error = abs(target_q - pred_q)
            
            if error < smallest_error:
                smallest_error = error
                best_choke = move
                
        # 2. FAILSAFE: If no safe moves exist, close the valve
        if best_choke is None:
            return max(0.0, current_choke - self.MAX_RAMP_RATE)
            
        return best_choke