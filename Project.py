import pandas as pd
import numpy as np

dates = pd.date_range('2013-01-01', periods=1000, freq='D')
sample_df = pd.DataFrame({
    'date': dates,
    'store_nbr': 1,
    'family': 'GROCERY I',
    'sales': 800 + np.sin(np.arange(1000)/365*2*np.pi)*200 + np.random.normal(0, 50, 1000),
    'onpromotion': np.random.choice([0,1], 1000)
})

print(sample_df.head())
print(f"Shape: {sample_df.shape}")
sample_df.to_csv('train_sample.csv', index=False)
print("Saved realistic train_sample.csv (1000 days with seasonality)")
