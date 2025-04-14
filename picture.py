import matplotlib.pyplot as plt
import numpy as np

# --- Actual Data Points from the Latest Log ---
iterations =        [0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10]
undefeated_rates =  [0.85, 0.95, 0.95, 0.90, 0.70, 0.95, 0.80, 0.85, 0.75, 0.95, 0.85]
# Status for iterations 1-10 based on [ACCEPT/REJECT NEW MODEL] log entry
# Iteration 0 has no status ('N/A' or similar)
status =            ['Initial', 'Accept', 'Reject', 'Reject', 'Reject', 'Reject', 'Reject', 'Reject', 'Reject', 'Reject', 'Reject']

# --- Assign colors based on status ---
colors = []
for s in status:
    if s == 'Accept':
        colors.append('green') # Green for accepted model evaluations
    elif s == 'Reject':
        colors.append('red')   # Red for rejected model evaluations
    else: # Initial
        colors.append('blue')  # Blue for the initial evaluation

# --- Plotting ---
plt.figure(figsize=(12, 7))

# Optional: Plot a light grey line connecting all points first
plt.plot(iterations, undefeated_rates, linestyle='-', color='lightgray', zorder=1)

# Plot scatter points with different colors
# Use zorder=2 to ensure points are drawn on top of the line
plt.scatter(iterations, undefeated_rates, c=colors, s=60, zorder=2) # s is marker size

# Add titles and labels in English
plt.title('AlphaZero Actual Undefeated Rate vs Baseline per Iteration')
plt.xlabel('Training Iteration')
plt.ylabel('Undefeated Rate vs Baseline ((W+D)/Total)')

# Set Y-axis limits
plt.ylim(0.0, 1.00) # Adjust Y limits

# Set X-axis ticks and limits
plt.xticks(iterations)
plt.xlim(-0.5, 10.5)

# Add grid
plt.grid(True)

# --- Create custom legend ---
# Create proxy artists for legend entries
blue_dot = plt.Line2D([0], [0], marker='o', color='w', label='Initial Evaluation', markersize=8, markerfacecolor='blue')
green_dot = plt.Line2D([0], [0], marker='o', color='w', label='Accepted Model Eval', markersize=8, markerfacecolor='green')
red_dot = plt.Line2D([0], [0], marker='o', color='w', label='Rejected Model Eval', markersize=8, markerfacecolor='red')
plt.legend(handles=[blue_dot, green_dot, red_dot])
# --- End custom legend ---

# Display the plot
plt.show()

print("Plotting code updated to color-code points based on Accept/Reject status.")