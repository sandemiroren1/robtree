import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Function for plotting instances for debugging.
# Points and Y
X = []
Y= []
epsilon = 0.1
# Color mapping
colors = {0: 'green', 1: 'blue'}

fig, ax = plt.subplots(figsize=(6, 6))

for (x, y), label in zip(X, Y):
    # Plot the point with color based on label
    ax.plot(x, y, 'o', color=colors[label], label=f'Label {label}')
    
    # Draw the square of side 0.1 centered at (x, y)
    square = patches.Rectangle(
        (x - epsilon, y - epsilon),  # bottom-left corner
        2*epsilon,  # width
        2*epsilon,  # height
        linewidth=1,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(square)

# Avoid duplicate Y in legend
handles, labels_text = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels_text, handles))
ax.legend(unique_labels.values(), unique_labels.keys())

# Customize the plot
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_aspect('equal')  # Square aspect ratio
ax.grid(True)
plt.show()