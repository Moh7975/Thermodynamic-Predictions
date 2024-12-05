import CoolProp.CoolProp as CP  # Thermodynamic Tables
import numpy as np
import pandas as pd

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

# Define a range of pressures (in MPa) and temperatures (in °C)
pressures = np.linspace(0.01, 22.064, 50)  # Pressures from 0.01 MPa to critical pressure of water
temperatures = np.linspace(0.1, 374, 100)   # Temperatures from 0 °C to critical temperature of water

# Generate data
data = data_generation(pressures, temperatures)

# Convert data to a Pandas DataFrame
columns = ['Pressure (MPa)', 'Temperature (°C)', 'Specific Volume (m³/kg)', 
           'Enthalpy (kJ/kg)', 'Entropy (kJ/kg·K)', 'Internal Energy (kJ/kg)', 
           'Viscosity (μPa·s)', 'Thermal Conductivity (mW/m·K)', 
           'Density (kg/m³)', 'Prandtl Number (dimensionless)']

df = pd.DataFrame(data, columns=columns)

# Save the data to a CSV file
output_file = 'steam_table_pressure_temperature.csv'
df.to_csv(output_file, index=False)

# Display the DataFrame
print(df)
