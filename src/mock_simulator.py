import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

def find_csv_file():
    """Dynamically locates any dataset CSV file in nearby directories."""
    search_paths = [
        os.path.join("..", "data", "*.csv"),
        os.path.join("data", "*.csv"),
        "*.csv"
    ]
    
    for path in search_paths:
        matches = glob.glob(path)
        if matches:
            print(f"Found dataset at: {matches[0]}")
            return matches[0]
            
    raise FileNotFoundError("Could not locate any .csv file in ../data/, ./data/, or current directory.")

class MockSimulator:
    def __init__(self, csv_path=None):
        """Initializes the mock simulator by training on the found CSV data."""
        if csv_path is None:
            csv_path = find_csv_file()
            
        self.csv_path = csv_path
        self._train_models()
        
        # Initialize starting states (Safe defaults matching the start of dataset)
        self.current_choke = 30.0
        self.current_q = 90.0
        self.current_whp = 250.0
        self.current_flp = 180.0
        self.current_bhp = 3000.0

    def _train_models(self):
        """Trains Linear Regression models on the CSV data."""
        df = pd.read_csv(self.csv_path)
            
        df['Prev_WHP'] = df['WHP_psi'].shift(1)
        df['Prev_FLP'] = df['FLP_psi'].shift(1)
        df['Prev_BHP'] = df['BHP_psi'].shift(1)
        df['Prev_Q'] = df['OilRate_bbl_hr'].shift(1)
        
        df_clean = df.dropna()
        
        X = df_clean[['Choke_pct', 'Prev_Q', 'Prev_WHP', 'Prev_FLP', 'Prev_BHP']]
        
        self.model_Q = LinearRegression().fit(X, df_clean['OilRate_bbl_hr'])
        self.model_WHP = LinearRegression().fit(X, df_clean['WHP_psi'])
        self.model_FLP = LinearRegression().fit(X, df_clean['FLP_psi'])
        self.model_BHP = LinearRegression().fit(X, df_clean['BHP_psi'])
        
        print("Mock Simulator Engine Initialized! AI models trained successfully.")

    def step(self, new_choke_position):
        """Executes a 1-hour simulation step."""
        self.current_choke = np.clip(new_choke_position, 0.0, 100.0)
        
        X_input = pd.DataFrame([[
            self.current_choke, 
            self.current_q, 
            self.current_whp, 
            self.current_flp, 
            self.current_bhp
        ]], columns=['Choke_pct', 'Prev_Q', 'Prev_WHP', 'Prev_FLP', 'Prev_BHP'])
        
        next_q = self.model_Q.predict(X_input)[0]
        next_whp = self.model_WHP.predict(X_input)[0]
        next_flp = self.model_FLP.predict(X_input)[0]
        next_bhp = self.model_BHP.predict(X_input)[0]
        
        self.current_q = next_q + np.random.normal(0, 0.2)
        self.current_whp = next_whp + np.random.normal(0, 0.5)
        self.current_flp = next_flp + np.random.normal(0, 0.2)
        self.current_bhp = next_bhp + np.random.normal(0, 1.0)
        
        return self.current_q, self.current_whp, self.current_flp, self.current_bhp

if __name__ == "__main__":
    sim = MockSimulator()
    
    print(f"\nInitial Safe State  -> Flow (Q): {sim.current_q:.2f} bbl/hr | WHP: {sim.current_whp:.2f} psi")
    
    print("\n--- Applying Choke 35% for 1 hour ---")
    q, whp, flp, bhp = sim.step(35.0)
    print(f"Resulting State     -> Flow (Q): {q:.2f} bbl/hr | WHP: {whp:.2f} psi | FLP: {flp:.2f} psi | BHP: {bhp:.2f} psi")
    
    print("\n--- Applying Choke 40% for 1 hour ---")
    q, whp, flp, bhp = sim.step(40.0)
    print(f"Resulting State     -> Flow (Q): {q:.2f} bbl/hr | WHP: {whp:.2f} psi | FLP: {flp:.2f} psi | BHP: {bhp:.2f} psi")