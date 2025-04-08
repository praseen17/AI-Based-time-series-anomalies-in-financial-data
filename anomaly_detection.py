import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# 1. User Input (Optional)
def get_user_data():
    try:
        revenue = input("Enter 12 Revenue values separated by commas (or press Enter to use default): ").strip()
        net_income = input("Enter 12 Net Income values separated by commas (or press Enter to use default): ").strip()
        if revenue and net_income:
            revenue = list(map(float, revenue.split(',')))
            net_income = list(map(float, net_income.split(',')))
            if len(revenue) == 12 and len(net_income) == 12:
                return revenue, net_income
            else:
                print("Invalid input length. Using default values.")
    except:
        print("Invalid input format. Using default values.")
    return None, None

# 2. Load Data
user_revenue, user_net_income = get_user_data()

revenue = user_revenue if user_revenue else [250, 270, 260, 275, 300, 280, 260, 270, 265, 1000, 275, 285]
net_income = user_net_income if user_net_income else [90, 95, 92, 100, 105, 98, 96, 94, 97, 500, 100, 102]

df = pd.DataFrame({
    'Month': pd.date_range(start='2021-01-01', periods=12, freq='M'),
    'Revenue': revenue,
    'Net_Income': net_income
}).set_index('Month')

# 3. Normalize
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# 4. Z-Score
def z_score(df, threshold=1):
    z = np.abs((df - df.mean()) / df.std())
    return (z > threshold).any(axis=1).astype(int)

# 5. Isolation Forest
def isolation_forest(df):
    model = IsolationForest(contamination=0.1, random_state=42)
    return pd.Series(model.fit_predict(df)).map({1: 0, -1: 1}).values

# 6. LSTM Autoencoder
def lstm_autoencoder(df):
    X = np.expand_dims(df.values, axis=0)
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        RepeatVector(X.shape[1]),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(X.shape[2]))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=300, verbose=0)

    X_pred = model.predict(X)
    mse = np.mean(np.square(X_pred - X), axis=2).flatten()
    threshold = np.mean(mse) + 2 * np.std(mse)
    return (mse > threshold).astype(int)

# 7. Run Methods
methods = {
    "Z-Score": z_score(df_scaled),
    "Isolation Forest": isolation_forest(df_scaled),
    "LSTM Autoencoder": lstm_autoencoder(df_scaled)
}

# 8. Output & Plot
for name, preds in methods.items():
    anomaly_dates = df.index[np.where(preds == 1)]
    print(f"\n{name} detected anomalies at: {list(anomaly_dates)}")

    plt.figure(figsize=(10, 4))
    plt.plot(df_scaled['Revenue'], label='Revenue')
    plt.plot(df_scaled['Net_Income'], label='Net Income')
    plt.scatter(anomaly_dates, df_scaled.loc[anomaly_dates, 'Revenue'], color='red', label='Anomalies', zorder=5)
    plt.title(f"{name} - Detected Anomalies")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()