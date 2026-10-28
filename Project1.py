import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping

# %% 1. Load and Inspect Data
filename = 'nonlinear_mlp_data_seed_17772130.csv'
seed = 17772130
print(f"Data File: {filename}")
print(f"Group Seed: {seed}\n")

data = pd.read_csv(filename)

# Make all column headers uppercase to match the script's variables
data.columns = [col.upper() for col in data.columns] 

print('--- First 5 rows of data ---')
print(data.head(), '\n')

# Report rows, columns, data types, and missing values
print('--- Dataset Info (Rows, Columns, Data Types) ---')
print(data.info(), '\n')

print('--- Missing Values ---')
print(data.isna().sum(), '\n')

# Check for Duplicate Rows
num_total_rows = len(data)
num_unique_rows = len(data.drop_duplicates())
num_duplicates = num_total_rows - num_unique_rows
print(f'Number of duplicate rows: {num_duplicates}')

if num_duplicates > 0:
    print('Duplicate rows exist in the dataset.')
else:
    print('No duplicate rows found.')

# Check for Anomalous/Outlier Values (Z-score method)
z_scores = np.abs(stats.zscore(data))
outlier_rows, outlier_cols = np.where(z_scores > 3)
print(f'\nNumber of anomalous values (|Z-score| > 3): {len(outlier_rows)}')

if len(outlier_rows) > 0:
    print('Anomalous values detected at (row, col):')
    for r, c in zip(outlier_rows, outlier_cols):
        print(f'({r}, {c})')
else:
    print('No extreme anomalous values found based on Z-score method.')

# %% 2. Data Exploration (Scatter Plots)
print('\n--- Data Summary ---')
print(data.describe(), '\n')

plt.figure(figsize=(15, 5))
plt.suptitle('Data Exploration')

plt.subplot(1, 3, 1)
plt.plot(data['X'], data['U'], 'b.')
plt.xlabel('X')
plt.ylabel('U')
plt.title('U vs X')

plt.subplot(1, 3, 2)
plt.plot(data['Y'], data['U'], 'g.')
plt.xlabel('Y')
plt.ylabel('U')
plt.title('U vs Y')

plt.subplot(1, 3, 3)
plt.plot(data['Z'], data['U'], 'r.')
plt.xlabel('Z')
plt.ylabel('U')
plt.title('U vs Z')

plt.tight_layout()
plt.show()

# %% 3. Setup Division of Data for Training, Validation, Testing
X = data[['X', 'Y', 'Z']].values
y = data['U'].values

# Split: 70% Train, 15% Validation, 15% Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=seed)

# Standardize features (fitted only on Training data to prevent Data Leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %% 4 & 5. Setup 5 Configurations (Hyperparameter Tuning)
# Providing 3 different hidden architectures, 2 activations, and varying learning rate/batch size.
configs = [
    {'name': 'M1', 'hidden': [16, 8], 'activation': 'relu', 'optimizer': Adam(learning_rate=0.01), 'batch_size': 32},
    {'name': 'M2', 'hidden': [32, 16], 'activation': 'tanh', 'optimizer': Adam(learning_rate=0.01), 'batch_size': 32},
    {'name': 'M3', 'hidden': [32, 16, 8], 'activation': 'relu', 'optimizer': SGD(learning_rate=0.01, momentum=0.9), 'batch_size': 32},
    {'name': 'M4', 'hidden': [16, 8], 'activation': 'relu', 'optimizer': SGD(learning_rate=0.001, momentum=0.9), 'batch_size': 32},
    {'name': 'M5', 'hidden': [10], 'activation': 'tanh', 'optimizer': Adam(learning_rate=0.01), 'batch_size': 64}
]

best_val_perf = float('inf')
best_model = None
best_history = None
best_name = ''

print('--- Starting Training for 5 Configurations ---')
for conf in configs:
    print(f"\nTraining {conf['name']}...")
    
    # Create the Sequential model
    model = Sequential()
    model.add(Dense(conf['hidden'][0], input_dim=3, activation=conf['activation']))
    for units in conf['hidden'][1:]:
        model.add(Dense(units, activation=conf['activation']))
    model.add(Dense(1, activation='linear')) # Output layer for regression
    
    model.compile(optimizer=conf['optimizer'], loss='mse')
    
    # Early Stopping setup (stops up to 100 epochs max if validation fails to improve)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Train the network
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=conf['batch_size'],
        callbacks=[early_stop],
        verbose=0
    )
    
    # Calculate Validation Performance
    val_mse = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"  -> {conf['name']} Validation MSE: {val_mse:.6f}")
    
    if val_mse < best_val_perf:
        best_val_perf = val_mse
        best_model = model
        best_history = history
        best_name = conf['name']

print(f"\n*** Best Model Selected: {best_name} with Val MSE: {best_val_perf:.6f} ***\n")

# %% 6. Test the Network (Best Model)
# Evaluate ONLY on Test Set
test_predictions = best_model.predict(X_test_scaled).flatten()

test_mse = mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)

print(f'--- Final Evaluation on Test Set ({best_name}) ---')
print(f'Test MSE: {test_mse:.6f}')
print(f'Test MAE: {test_mae:.6f}')
print(f'Test RMSE: {test_rmse:.6f}\n')

# Plot Regression for Actual vs Predicted & Learning Curves
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(y_test, test_predictions, 'mo', markerfacecolor='m')
p_min = min(np.min(y_test), np.min(test_predictions))
p_max = max(np.max(y_test), np.max(test_predictions))
plt.plot([p_min, p_max], [p_min, p_max], 'k--', lw=2)
plt.xlabel('Actual U')
plt.ylabel('Predicted U')
plt.title('Test Data: Actual vs Predicted')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'], label='Train Loss (MSE)')
plt.plot(best_history.history['val_loss'], label='Val Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title(f'{best_name} Training History')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% 7. Bonus: Controlled Sweep
# Fix two variables to their mean, sweep the third
mean_x = data['X'].mean()
mean_y = data['Y'].mean()
mean_z = data['Z'].mean()

# Sweep Z (keep X and Y at mean)
z_sweep = np.linspace(-2, 6, 100)
sweep_inputs_z = np.column_stack([np.full(100, mean_x), np.full(100, mean_y), z_sweep])
sweep_inputs_z_scaled = scaler.transform(sweep_inputs_z)
sweep_outputs_z = best_model.predict(sweep_inputs_z_scaled).flatten()

# Sweep X (keep Y and Z at mean)
x_sweep = np.linspace(0.5, 3.5, 100)
sweep_inputs_x = np.column_stack([x_sweep, np.full(100, mean_y), np.full(100, mean_z)])
sweep_inputs_x_scaled = scaler.transform(sweep_inputs_x)
sweep_outputs_x = best_model.predict(sweep_inputs_x_scaled).flatten()

plt.figure(figsize=(14, 5))
plt.suptitle('Bonus: Controlled Sweep')

plt.subplot(1, 2, 1)
plt.plot(z_sweep, sweep_outputs_z, 'r-', lw=2)
plt.xlabel('Z (Sweep)')
plt.ylabel('Predicted U')
plt.title(f'Sweep Z (X={mean_x:.2f}, Y={mean_y:.2f})')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_sweep, sweep_outputs_x, 'b-', lw=2)
plt.xlabel('X (Sweep)')
plt.ylabel('Predicted U')
plt.title(f'Sweep X (Y={mean_y:.2f}, Z={mean_z:.2f})')
plt.grid(True)

plt.tight_layout()
plt.show()

