import os
import CoolProp.CoolProp as CP  # Thermodynamic Tables
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor  # Using Random Forest for regression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


pressures = np.linspace(0.01, 1000, 50)
temperatures = np.linspace(0.1, 800, 100) 
    
# Function to generate steam table data for a range of pressures and temperatures
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
    # If the file does not exist, generate the data
    data = data_generation(pressures, temperatures)

    columns = ['Pressure (MPa)', 'Temperature (°C)', 'Specific Volume (m³/kg)', 
               'Enthalpy (kJ/kg)', 'Entropy (kJ/kg·K)', 'Internal Energy (kJ/kg)', 
               'Viscosity (μPa·s)', 'Thermal Conductivity (mW/m·K)', 
               'Density (kg/m³)', 'Prandtl Number (dimensionless)']

    df = pd.DataFrame(data, columns=columns)

    # Remove redundant rows
    df = df.drop_duplicates(subset=['Pressure (MPa)', 'Temperature (°C)'])
    
    # Save the data to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Data generated, deduplicated, and saved to file: {output_file}")

# Define the features (X) and target variables (y)
X = df[['Pressure (MPa)', 'Temperature (°C)']].values  # Features: Pressure and Temperature
y = df.iloc[:, 2:].values  # Target variables: All other properties

# Split the data into training and test sets
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.3, random_state=3, shuffle=True)

# Verify the shapes of the splits
print(f"Xtr shape: {Xtr.shape}")
print(f"Xts shape: {Xts.shape}")
print(f"ytr shape: {ytr.shape}")
print(f"yts shape: {yts.shape}")


# Train a multi-output regression model
model = MultiOutputRegressor(RandomForestRegressor(random_state=3))
model.fit(Xtr, ytr)

# Predict on the test set
ypred = model.predict(Xts)

mse = mean_squared_error(yts, ypred, multioutput='raw_values')
print("Mean Squared Error for each target variable:", mse)



# Plot viscosity vs. temperature at different pressures
def plot_viscosity_vs_temperature(df, pressures_to_plot, round_precision=2):
    # Round the Pressure column to the desired precision
    df['Rounded Pressure (MPa)'] = df['Pressure (MPa)'].round(round_precision)
    
    plt.figure(figsize=(10, 6))
    for pressure in pressures_to_plot:
        # Round the pressure to match the dataset's precision
        rounded_pressure = round(pressure, round_precision)
        data_at_pressure = df[df['Rounded Pressure (MPa)'] == rounded_pressure]
        
        if not data_at_pressure.empty:  # Check if the filtered data is not empty
            plt.plot(data_at_pressure['Temperature (°C)'], 
                     data_at_pressure['Viscosity (μPa·s)'], 
                     label=f'{pressure} MPa')
        else:
            print(f"No data found for Pressure = {pressure} MPa")
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Viscosity (μPa·s)')
    plt.title('Viscosity vs Temperature at Different Pressures')
    plt.legend()
    plt.grid(True)
    plt.show()


selected_pressures = pressures[::5]  # Select every 5th value (50 / 10 = 5)
plot_viscosity_vs_temperature(df, selected_pressures)

def plot_viscosity_comparison(Xts, yts, ypred):
    temperatures = Xts[:, 1]  # Extract temperatures from test features
    actual_viscosity = yts[:, 5]  # Actual viscosity is the 6th column in the target
    predicted_viscosity = ypred[:, 5]  # Predicted viscosity is also in the 6th column

    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures, actual_viscosity, color='blue', label='Actual Viscosity', alpha=0.5, s=10)
    plt.scatter(temperatures, predicted_viscosity, color='red', label='Predicted Viscosity', alpha=0.5, s=10)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Viscosity (μPa·s)')
    plt.title('Actual vs Predicted Viscosity vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot
plot_viscosity_comparison(Xts, yts, ypred)

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
plot_viscosity_comparison_at_pressure(Xts, yts, ypred, pressure=979.5920408163265)


def plot_targets_comparison_at_pressure(Xts, yts, ypred, pressure, target_labels):
    """
    Plots all targeted values (e.g., Specific Volume, Enthalpy, etc.) at a specific pressure.

    Parameters:
    Xts: array-like
        Test set features.
    yts: array-like
        Actual target values in the test set.
    ypred: array-like
        Predicted target values in the test set.
    pressure: float
        The specific pressure at which to filter the data.
    target_labels: list
        List of target column names for labeling the plots.
    """
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
        
        
    # Define the target labels (matching the column order in yts)
target_labels = [
    'Specific Volume (m³/kg)', 'Enthalpy (kJ/kg)', 'Entropy (kJ/kg·K)', 
    'Internal Energy (kJ/kg)', 'Viscosity (μPa·s)', 'Thermal Conductivity (mW/m·K)', 
    'Density (kg/m³)', 'Prandtl Number (dimensionless)'
]


# Call the function to plot all targets for a specific pressure
plot_targets_comparison_at_pressure(Xts, yts, ypred, pressure=979.5920408163265, target_labels=target_labels)