import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Road conditions and their impact on accident probability
road_conditions = ['dry', 'wet', 'icy', 'gravel', 'muddy']
road_condition_weights = [0.5, 0.2, 0.1, 0.1, 0.1]

# Severity levels
severity_levels = ['none', 'low', 'medium', 'high']

# Function to simulate one row
def generate_row():
    speed = np.random.normal(loc=60, scale=15)  # average speed 60 km/h
    speed = max(0, min(160, speed))  # limit to 0-160
    acceleration = np.random.uniform(0.5, 6.0)
    impact_force = np.random.uniform(0.0, 10.0)
    road = random.choices(road_conditions, weights=road_condition_weights)[0]

    # Simple accident logic
    accident = 0
    severity = 'none'
    if impact_force > 7 or (acceleration > 4.5 and speed > 80) or road in ['icy', 'gravel'] and impact_force > 5:
        accident = 1
        if impact_force > 9:
            severity = 'high'
        elif impact_force > 6:
            severity = 'medium'
        else:
            severity = 'low'

    return [round(speed, 2), round(acceleration, 2), round(impact_force, 2), road, accident, severity]

# Generate dataset
data = [generate_row() for _ in range(20000)]

# Create DataFrame
df = pd.DataFrame(data, columns=[
    'speed', 'acceleration', 'impact_force', 'road_condition', 'is_accident', 'severity'
])

# Save to CSV
df.to_csv('sensor_data.csv', index=False)
print("✅ Dataset generated: sensor_data.csv with 20,000 rows")
