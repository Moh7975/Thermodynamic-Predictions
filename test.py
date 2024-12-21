import os
import CoolProp.CoolProp as CP  # Thermodynamic Tables
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor  # Using Random Forest for regression
import matplotlib.pyplot as plt

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
    pressures = np.linspace(0.01, 22.064, 50)  # Pressures from 0.01 MPa to critical pressure of water
    temperatures = np.linspace(0.1, 374, 100)  # Temperatures from 0 °C to critical temperature of water

    # Generate data
    data = data_generation(pressures, temperatures)

    # Convert data to a Pandas DataFrame
    columns = ['Pressure (MPa)', 'Temperature (°C)', 'Specific Volume (m³/kg)', 
               'Enthalpy (kJ/kg)', 'Entropy (kJ/kg·K)', 'Internal Energy (kJ/kg)', 
               'Viscosity (μPa·s)', 'Thermal Conductivity (mW/m·K)', 
               'Density (kg/m³)', 'Prandtl Number (dimensionless)']

    df = pd.DataFrame(data, columns=columns)

    # Save the data to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Data generated and saved to file: {output_file}")

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

# Evaluate the model
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(yts, ypred, multioutput='raw_values')
print("Mean Squared Error for each target variable:", mse)

# Plot viscosity vs. temperature at different pressures
def plot_viscosity_vs_temperature(df, pressures_to_plot):
    plt.figure(figsize=(10, 6))
    for pressure in pressures_to_plot:
        data_at_pressure = df[df['Pressure (MPa)'] == pressure]
        plt.plot(data_at_pressure['Temperature (°C)'], 
                 data_at_pressure['Viscosity (μPa·s)'], 
                 label=f'{pressure} MPa')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Viscosity (μPa·s)')
    plt.title('Viscosity vs Temperature at Different Pressures')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define pressures to plot
pressures_to_plot = [0.1, 1, 5, 10, 20]  # Example pressures in MPa
plot_viscosity_vs_temperature(df, pressures_to_plot)

def plot_viscosity_comparison(Xts, yts, ypred):
    temperatures = Xts[:, 1]  # Extract temperatures from test features
    actual_viscosity = yts[:, 5]  # Actual viscosity is the 6th column in the target
    predicted_viscosity = ypred[:, 5]  # Predicted viscosity is also in the 6th column

    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures, actual_viscosity, color='blue', label='Actual Viscosity', alpha=0.6)
    plt.scatter(temperatures, predicted_viscosity, color='red', label='Predicted Viscosity', alpha=0.6)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Viscosity (μPa·s)')
    plt.title('Actual vs Predicted Viscosity vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot
plot_viscosity_comparison(Xts, yts, ypred)