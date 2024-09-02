
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_court(ax=None):
    """Plots the basketball court on a given axis."""
    if ax is None:
        ax = plt.gca()
    
    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    ax.set_aspect('equal')
    
    court_elements = [
        plt.Circle((0, 0), radius=7.5, linewidth=2, color='black', fill=False),  # Hoop
        plt.Rectangle((-30, -10), 60, -1, linewidth=2, color='black'),  # Backboard
        plt.Rectangle((-80, -47.5), 160, 190, linewidth=2, color='black', fill=False),  # Paint
        plt.Circle((0, 142.5), radius=60, linewidth=2, color='black', fill=False),  # Free throw top arc
        plt.Circle((0, 142.5), radius=60, linewidth=2, color='black', fill=False, linestyle='dashed'),  # Free throw bottom arc
        patches.Arc((0, 0), 475, 475, theta1=0, theta2=180, linewidth=2, color='black'),  # 3-point arc
        plt.Rectangle((-250, -47.5), 500, 470, linewidth=2, color='black', fill=False),  # Outer lines
    ]
    
    for element in court_elements:
        ax.add_patch(element)
    
    return ax

def plot_shot_chart_hexbin(shots, title, opponent, court_color='white'):
    """Plots a hexbin shot chart."""
    plt.figure(figsize=(12, 11))
    ax = plt.gca()
    ax.set_facecolor(court_color)
    plot_court(ax)
    
    hexbin = plt.hexbin(
        shots['LOC_X'], shots['LOC_Y'], C=shots['SHOT_MADE_FLAG'], 
        gridsize=40, extent=(-250, 250, -47.5, 422.5), cmap='Blues', edgecolors='grey'
    )
    
    cb = plt.colorbar(hexbin, ax=ax, orientation='vertical')
    cb.set_label('Shooting Percentage')
    
    total_attempts = len(shots)
    total_made = shots['SHOT_MADE_FLAG'].sum()
    overall_percentage = total_made / total_attempts if total_attempts > 0 else 0
    
    opponent_text = f" against {opponent}" if opponent else " against the rest of the league"
    
    plt.text(0, 450, f"Total Shots: {total_attempts}", fontsize=12, ha='center')
    plt.text(0, 430, f"Total Made: {total_made}", fontsize=12, ha='center')
    plt.text(0, 410, f"Overall Percentage: {overall_percentage:.2%}", fontsize=12, ha='center')
    
    plt.title(f"{title}{opponent_text}", pad=50)  # Adjusted title to include opponent
    plt.xlim(-250, 250)
    plt.ylim(-47.5, 422.5)
    
    plt.tight_layout()  # Ensures that the plot elements don't overlap
    return plt.gcf()

