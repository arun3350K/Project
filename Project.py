import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.preprocessing import StandardScaler

# Step 1: Load California Tobacco Control Data (Proposition 99, 1988)
# Download from: https://raw.githubusercontent.com/bcastillo/synthcontrol/master/data/smoking.csv
url = "https://raw.githubusercontent.com/bcastillo/synthcontrol/master/data/smoking.csv"
data = pd.read_csv(url, index_col=0)

print("Data shape:", data.shape)
print("Columns (states):", data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

# Treated unit: California (first column), Intervention: 1988 Q4 (time 35, 0-indexed)
treated_unit = 'California'
intervention_time = 35  # Quarter when Prop 99 passed
pre_period = data.index < intervention_time
post_period = data.index >= intervention_time

# Step 2: Synthetic Control Class (Optimized via Quadratic Programming)
class SyntheticControl:
    def __init__(self):
        self.weights_ = None
        self.rmspe_pre_ = None
        
    def fit(self, Y, treated_idx, pre_period_mask):
        """Fit synthetic control using pre-treatment periods"""
        Y_pre = Y.loc[pre_period_mask].values  # Pre-treatment matrix (T_pre x J+1)
        y_treated = Y_pre[:, treated_idx]      # Treated outcome pre-treatment
        X_pool = np.delete(Y_pre, treated_idx, axis=1)  # Donor pool pre-treatment
        
        # Optimization: Minimize ||Xw - y||^2 s.t. w >= 0, sum(w) = 1
        w = cp.Variable(X_pool.shape[1])
        objective = cp.Minimize(cp.sum_squares(X_pool @ w - y_treated))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        self.weights_ = w.value
        self.Y_ = Y.values
        self.treated_idx_ = treated_idx
        
        # Pre-treatment RMSPE (good fit if low, e.g., <0.05)
        synth_pre = X_pool @ self.weights_
        self.rmspe_pre_ = np.sqrt(np.mean((synth_pre - y_treated)**2))
        print(f"Pre-treatment RMSPE: {self.rmspe_pre_:.4f}")
        
        return self
    
    def predict(self, Y):
        """Generate synthetic control for full period"""
        donor_pool = np.delete(Y.values, self.treated_idx_, axis=1)
        return donor_pool @ self.weights_
    
    def plot(self, title="SCM: Actual vs Synthetic"):
        Y_synth = self.predict(self.Y_)
        treated_actual = self.Y_[:, self.treated_idx_]
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.Y_.index, treated_actual, 'b-', label='California (Treated)', linewidth=2)
        plt.plot(self.Y_.index, Y_synth, 'r--', label='Synthetic Control', linewidth=2)
        plt.axvline(x=data.index[intervention_time], color='k', linestyle=':', label='Intervention (1988 Q4)')
        plt.fill_between(self.Y_.index, treated_actual, Y_synth, alpha=0.3, color='gray', label='Treatment Gap')
        plt.ylabel('Log Cigarette Sales per Capita')
        plt.xlabel('Time (Quarterly)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Step 3: Fit SCM Model
treated_idx = list(data.columns).index(treated_unit)
model = SyntheticControl()
model.fit(data, treated_idx, pre_period)

print("\nDonor Weights (top 5):")
weights_df = pd.DataFrame({
    'State': [col for col in data.columns if col != treated_unit],
    'Weight': model.weights_
}).sort_values('Weight', ascending=False)
print(weights_df.head())

# Step 4: Calculate ATT (Average Treatment Effect on Treated)
synth_full = model.predict(data)
att_post = data[treated_unit][post_period] - synth_full[post_period]
avg_att = att_post.mean()
print(f"\nPost-Intervention ATT: {avg_att:.3f} (negative = policy reduced sales)")
print(f"ATT Std: {att_post.std():.3f}")

# Step 5: Placebo/In-Space Test (Robustness)
def placebo_test(data, model_class, treated_idx, intervention_time, n_placebos=10):
    """Run SCM treating each donor as 'treated' (placebo)"""
    placebos = []
    donor_states = [i for i in range(data.shape[1]) if i != treated_idx]
    
    for placebo_idx in donor_states[:n_placebos]:
        placebo_model = model_class()
        placebo_model.fit(data, placebo_idx, data.index < intervention_time)
        synth_placebo = placebo_model.predict(data)
        placebo_att = data.iloc[post_period, placebo_idx].mean() - synth_placebo[post_period].mean()
        placebos.append(placebo_att)
    
    return np.array(placebos)

placebo_atts = placebo_test(data, SyntheticControl, treated_idx, intervention_time)
p_value = np.mean(np.abs(placebo_atts) >= np.abs(avg_att))
print(f"\nPlacebo p-value: {p_value:.3f} (significant if <0.1)")

# Plot placebo distribution
plt.figure(figsize=(10, 5))
plt.hist(placebo_atts, bins=10, alpha=0.7, label='Placebo ATT', color='orange')
plt.axvline(avg_att, color='red', linewidth=3, label=f'California ATT ({avg_att:.3f})')
plt.xlabel('ATT Estimate')
plt.ylabel('Frequency')
plt.title('Placebo Test: California Effect vs Donor States')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Step 6: Summary Results
print("\n" + "="*50)
print("SYNTHETIC CONTROL METHOD RESULTS")
print("="*50)
print(f"Treated Unit: {treated_unit}")
print(f"Intervention: 1988 Q4 (time {intervention_time})")
print(f"Pre-RMSPE: {model.rmspe_pre_:.4f}")
print(f"Post-ATT: {avg_att:.3f} packs/capita")
print(f"Placebo p-value: {p_value:.3f}")
print(f"Key Donors: {weights_df.head(3)['State'].tolist()}")
print("="*50)

model.plot()
plt.show()

print("\n✓ Complete SCM analysis: Model fit, ATT, placebo test, diagnostics complete!")
