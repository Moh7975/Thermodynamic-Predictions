import os
import CoolProp.CoolProp as CP  # Thermodynamic Tables
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pressures = np.linspace(1, 1250, 250) #Thermodynamic Tables has maximum range of pressure from 1 to 1250 MPa
#Cleaning Pressure values
pressures = np.round(pressures)
pressures = np.unique(pressures)
temperatures = np.linspace(1, 1000, 500)
columns = ['Pressure (MPa)', 'Temperature (°C)', 'Specific Volume (m³/kg)',
               'Enthalpy (kJ/kg)', 'Entropy (kJ/kg·K)', 'Internal Energy (kJ/kg)',
               'Viscosity (μPa·s)', 'Thermal Conductivity (mW/m·K)',
               'Density (kg/m³)', 'Prandtl Number (dimensionless)']
#Values to be Predicted
target_labels = [
    'Specific Volume (m³/kg)', 'Enthalpy (kJ/kg)', 'Entropy (kJ/kg·K)',
    'Internal Energy (kJ/kg)', 'Viscosity (μPa·s)', 'Thermal Conductivity (mW/m·K)',
    'Density (kg/m³)', 'Prandtl Number (dimensionless)'
]
selected_pressures = pressures[::5]  # Select every 5th value (50 / 10 = 5)

#Regressors used
regressors = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1, random_state=3),  # Regularization parameter (alpha) can be tuned
    "Polynomial Regression (Degree=3)": Pipeline([
        ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
        ('linear_regression', LinearRegression())
    ]),
    "Support Vector Regression": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=3),
    "Random Forest": RandomForestRegressor(random_state=3),
    "XGBoost": XGBRegressor(random_state=3),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Neural Network Regression": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=3)
}

results = {}


def data_generation(pressures, temperatures):
    data = []
    for P in pressures:
        for T in temperatures:
            try:
                # Calculate properties for the given pressure and temperature
                v = 1 / CP.PropsSI('D', 'P', P * 1e6, 'T', T + 273.15, 'Water')  # Specific volume (m³/kg)
                h = CP.PropsSI('H', 'P', P * 1e6, 'T', T + 273.15, 'Water') / 1000  # Enthalpy (kJ/kg)
                s = CP.PropsSI('S', 'P', P * 1e6, 'T', T + 273.15, 'Water') / 1000  # Entropy (kJ/kg·K)
                u = CP.PropsSI('U', 'P', P * 1e6, 'T', T + 273.15, 'Water') / 1000  # Internal energy (kJ/kg)
                eta = CP.PropsSI('VISCOSITY', 'P', P * 1e6, 'T', T + 273.15, 'Water') * 1e6  # Viscosity (μPa·s)
                lambd = CP.PropsSI('CONDUCTIVITY', 'P', P * 1e6, 'T', T + 273.15, 'Water') * 1e3  # Thermal conductivity (mW/m·K)
                rho = CP.PropsSI('D', 'P', P * 1e6, 'T', T + 273.15, 'Water')  # Density (kg/m³)
                Pr = eta * CP.PropsSI('C', 'P', P * 1e6, 'T', T + 273.15, 'Water') / lambd  # Prandtl number

                # Append data to the list
                data.append([P, T, v, h, s, u, eta, lambd, rho, Pr])
            except Exception as e:
                print(f"Error at P={P} MPa, T={T} °C: {e}")
                continue
    return data


# Define the output file path
output_file = 'steam_table_pressure_temperature.csv'

# Check if the file exists
if os.path.exists(output_file):
    # If the file exists, load it into a DataFrame
    df = pd.read_csv(output_file)
    print(f"Data loaded from existing file: {output_file}")
else:
    # Generate data
    data = data_generation(pressures, temperatures)
    df = pd.DataFrame(data, columns=columns)
    df = df.drop_duplicates(subset=['Pressure (MPa)', 'Temperature (°C)'])

    # Save the data to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Data generated, deduplicated, and saved to file: {output_file}")
    
    # Define the features (X) and target variables (y)
X = df[['Pressure (MPa)', 'Temperature (°C)']].values  # Features: Pressure and Temperature
y = df.iloc[:, 2:].values  # Target variables: All other properties

# Split the data into training and test sets
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.3, random_state=3, shuffle=True)

# Train a multi-output regression model
model = MultiOutputRegressor(RandomForestRegressor(random_state=3))
model.fit(Xtr, ytr)

# Predict on the test set
ypred = model.predict(Xts)


# Calculate the Mean Squared Error for each target variable
mse = mean_squared_error(yts, ypred, multioutput='raw_values')

# Display the MSE for each target in a readable format
print("Mean Squared Error for each target variable:")
for label, error in zip(target_labels, mse):
    print(f"{label}: {error:.4e}")  # Formats the error in scientific notation with 4 decimal places

def plot_actual_viscosity_vs_temperature_fixed(Xts, yts, pressures_to_plot):

    plt.figure(figsize=(12, 8))

    for pressure in pressures_to_plot:
        # Filter the data for the specific pressure
        indices_at_pressure = np.isclose(Xts[:, 0], pressure, atol=0.01)  # Check pressure in the first column of Xts
        temperatures = Xts[indices_at_pressure, 1]  # Extract temperatures for the filtered rows
        actual_viscosity = yts[indices_at_pressure, 5]  # Extract actual viscosity for the filtered rows

        if len(temperatures) > 0:  # Ensure there are data points to plot
            # Sort data by temperature to ensure smooth lines
            sorted_indices = np.argsort(temperatures)
            temperatures_sorted = temperatures[sorted_indices]
            viscosity_sorted = actual_viscosity[sorted_indices]

            # Plot the sorted data
            plt.plot(temperatures_sorted, viscosity_sorted, label=f'{pressure:.2f} MPa', marker='o', alpha=0.7)
        else:
            print(f"No data found for Pressure = {pressure} MPa")

    plt.xlabel('Temperature (°C)')
    plt.ylabel('Viscosity (μPa·s)')
    plt.title('Viscosity vs Temperature at Different Pressures (Actual Data)')
    plt.legend(title="Pressure (MPa)", loc='best')
    plt.grid(True)
    plt.show()

plot_actual_viscosity_vs_temperature_fixed(Xts, yts, selected_pressures)
def plot_viscosity_comparison_at_pressure(Xts, yts, ypred, pressure):
    # Filter the test data for the specified pressure
    indices_at_pressure = np.isclose(Xts[:, 0], pressure, atol=0.01)  # Check pressure in the first column of Xts
    temperatures = Xts[indices_at_pressure, 1]  # Extract temperatures for the filtered rows
    actual_viscosity = yts[indices_at_pressure, 5]  # Actual viscosity for the filtered rows
    predicted_viscosity = ypred[indices_at_pressure, 5]  # Predicted viscosity for the filtered rows

    if len(temperatures) > 0:  # Ensure there are data points to plot
        plt.figure(figsize=(10, 6))
        plt.scatter(temperatures, actual_viscosity, color='blue', label='Actual Viscosity', alpha=0.5, s=10, marker='o')
        plt.scatter(temperatures, predicted_viscosity, color='red', label='Predicted Viscosity', alpha=0.5, s=10, marker='x')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Viscosity (μPa·s)')
        plt.title(f'Actual vs Predicted Viscosity at Pressure = {pressure} MPa')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No test data found for Pressure = {pressure} MPa")
        
# Example: Compare actual and predicted viscosity for a specific pressure in the test set
plot_viscosity_comparison_at_pressure(Xts, yts, ypred, pressure=1021)   

def plot_targets_comparison_at_pressure(Xts, yts, ypred, pressure, target_labels):

    # Filter the test data for the specified pressure
    indices_at_pressure = np.isclose(Xts[:, 0], pressure, atol=0.01)  # Check pressure in the first column of Xts
    temperatures = Xts[indices_at_pressure, 1]  # Extract temperatures for the filtered rows

    if len(temperatures) > 0:  # Ensure there are data points to plot
        num_targets = yts.shape[1]  # Number of target columns
        plt.figure(figsize=(12, 8))

        for i in range(num_targets):
            actual_values = yts[indices_at_pressure, i]  # Actual values for the current target
            predicted_values = ypred[indices_at_pressure, i]  # Predicted values for the current target

            # Plot actual and predicted values for the current target
            plt.subplot((num_targets + 1) // 2, 2, i + 1)  # Create subplots in a grid format
            plt.scatter(temperatures, actual_values, color='blue', label='Actual', alpha=0.7, s=10, marker='o')
            plt.scatter(temperatures, predicted_values, color='red', label='Predicted', alpha=0.7, s=10, marker='x')
            plt.title(f'{target_labels[i]}')
            plt.xlabel('Temperature (°C)')
            plt.ylabel(target_labels[i])
            plt.legend()
            plt.grid(True)

        plt.tight_layout()  # Adjust layout for better visualization
        plt.suptitle(f'Actual vs Predicted Values at Pressure = {pressure} MPa', y=1.02, fontsize=16)
        plt.show()
    else:
        print(f"No test data found for Pressure = {pressure} MPa")

# Call the function to plot all targets for a specific pressure
plot_targets_comparison_at_pressure(Xts, yts, ypred, pressure=1148, target_labels=target_labels)

def plot_predicted_values_at_pressure_and_temperature_range_with_model(model, Xts, pressure, temp_min, temp_max, target_labels):

    # Generate new input data for the specified pressure and temperature range
    temperatures = np.linspace(temp_min, temp_max, 100)  # Generate 100 points in the temperature range
    pressures = np.full_like(temperatures, pressure)  # Fixed pressure value for all points
    X_new = np.column_stack((pressures, temperatures))  # Combine into feature array

    # Predict values using the trained model
    ypred_new = model.predict(X_new)

    if len(temperatures) > 0:  # Ensure there are data points to plot
        num_targets = ypred_new.shape[1]  # Number of target variables
        plt.figure(figsize=(12, 8))

        for i in range(num_targets):
            # Plot predicted values for the current target
            plt.subplot((num_targets + 1) // 2, 2, i + 1)  # Create subplots in a grid format
            plt.plot(temperatures, ypred_new[:, i], color='green', label='Predicted')
            plt.title(f'{target_labels[i]}')
            plt.xlabel('Temperature (°C)')
            plt.ylabel(target_labels[i])
            plt.legend()
            plt.grid(True)

        plt.tight_layout()  # Adjust layout for better visualization
        plt.suptitle(f'Predicted Values at Pressure = {pressure} MPa\nTemperature Range = {temp_min}–{temp_max} °C', y=1.02, fontsize=16)
        plt.show()
    else:
        print(f"No predicted data found for Pressure = {pressure} MPa in the range {temp_min}–{temp_max} °C")
        
        
    

plot_predicted_values_at_pressure_and_temperature_range_with_model(
    model, Xts, pressure=3000, temp_min=0, temp_max=1000, target_labels=target_labels
)


best_worst_results = {label: {"best_regressor": None, "best_mse": float('inf'),
                              "worst_regressor": None, "worst_mse": float('-inf')}
                      for label in target_labels}

for name, base_model in regressors.items():
    print(f"Training {name}...")
    model = MultiOutputRegressor(base_model)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xts)

    # Calculate the MSE for each target variable
    mse = mean_squared_error(yts, ypred, multioutput='raw_values')
    avg_mse = np.mean(mse)  # Calculate average MSE across all targets
    results[name] = {
        "model": model,
        "ypred": ypred,
        "mse": mse,
        "avg_mse": avg_mse,
    }

    # Print MSE for each target variable
    print(f"\n{name} - Mean Squared Error for each target:")
    for label, error in zip(target_labels, mse):
        print(f"{label}: {error:.4e}")

        # Update the best regressor for the target
        if error < best_worst_results[label]["best_mse"]:
            best_worst_results[label]["best_regressor"] = name
            best_worst_results[label]["best_mse"] = error

        # Update the worst regressor for the target
        if error > best_worst_results[label]["worst_mse"]:
            best_worst_results[label]["worst_regressor"] = name
            best_worst_results[label]["worst_mse"] = error

    print(f"Average MSE for {name}: {avg_mse:.4e}")
    print("-" * 40)

# Compare average MSE across all models
print("\nComparison of Average MSE Across Regressors:")
for name, result in results.items():
    print(f"{name}: Average MSE = {result['avg_mse']:.4e}")

# Print the best and worst regressors for each target variable
print("\nBest and Worst Regressors for Each Target Variable:")
for label in target_labels:
    print(f"{label}:")
    print(f"  Best Regressor: {best_worst_results[label]['best_regressor']} (MSE: {best_worst_results[label]['best_mse']:.4e})")
    print(f"  Worst Regressor: {best_worst_results[label]['worst_regressor']} (MSE: {best_worst_results[label]['worst_mse']:.4e})")
    print("-" * 40)

def plot_all_targets_comparison_for_regressors(results, Xts, yts, pressure, target_labels):

    # Filter the test data for the specified pressure
    indices_at_pressure = np.isclose(Xts[:, 0], pressure, atol=0.01)
    temperatures = Xts[indices_at_pressure, 1]  # Extract temperatures for the filtered rows

    if len(temperatures) > 0:
        # Sort the data by temperature for smooth lines
        sorted_indices = np.argsort(temperatures)
        temperatures_sorted = temperatures[sorted_indices]

        num_targets = yts.shape[1]  # Number of target variables
        plt.figure(figsize=(16, 12))

        for i in range(num_targets):
            # Sort actual values by temperature
            actual_values = yts[indices_at_pressure, i]
            actual_values_sorted = actual_values[sorted_indices]

            # Plot actual data
            plt.subplot((num_targets + 1) // 2, 2, i + 1)
            plt.plot(temperatures_sorted, actual_values_sorted, color='black', label='Actual', linewidth=2)

            # Plot predicted values for each regressor
            for name, result in results.items():
                ypred = result["ypred"]
                predicted_values = ypred[indices_at_pressure, i]
                predicted_values_sorted = predicted_values[sorted_indices]

                plt.plot(temperatures_sorted, predicted_values_sorted, label=f"{name} (Predicted)", alpha=0.8)

            plt.title(target_labels[i])
            plt.xlabel('Temperature (°C)')
            plt.ylabel(target_labels[i])
            plt.grid(True)

            # Position the legend outside the plot area
            if i % 2 == 0:  # For left-side subplots
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', frameon=False)
            else:  # For right-side subplots
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', frameon=False)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate legends
        plt.suptitle(f'Comparison of Predicted Values for All Targets at Pressure = {pressure} MPa', y=1.02, fontsize=16)
        plt.show()
    else:
        print(f"No test data found for Pressure = {pressure} MPa")

plot_all_targets_comparison_for_regressors(results, Xts, yts, pressure=1021,target_labels=target_labels)

plot_all_targets_comparison_for_regressors(results, Xts, yts, pressure=485.0,target_labels=target_labels)

plot_all_targets_comparison_for_regressors(results, Xts, yts, pressure=1,target_labels=target_labels)


