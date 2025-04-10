import numpy as np
from scipy.interpolate import interp1d

# Approximate data points from the blue line in the image (hour vs relative_power)
# You may need to refine these values based on more accurate digitization
hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
relative_power = np.array([0.08, 0.05, 0.02, 0.01, 0.15, 0.40, 0.48, 0.35, 0.15, 0.05, 0.02])

# Create interpolation function
interp_func = interp1d(hours, relative_power, kind='cubic', fill_value="extrapolate")

def get_relative_power(hour, minute):
    # Convert hour and minute to fractional hour
    time = hour + minute / 60.0
    # Interpolate relative_power
    power = float(interp_func(time))
    return power

# Example usage
hour = 11
minute = 30
value = get_relative_power(hour, minute)
print(f"Relative power at {hour}:{minute:02d} is approximately {value:.3f}")
