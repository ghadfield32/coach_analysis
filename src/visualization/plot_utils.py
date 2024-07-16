import matplotlib.pyplot as plt
import seaborn as sns

def plot_court(ax=None):
    if ax is None:
        ax = plt.gca()
    
    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    ax.set_aspect('equal')
    
    court_elements = [
        plt.Circle((0, 0), radius=7.5, linewidth=2, color='black', fill=False),
        plt.Rectangle((-30, -10), 60, -1, linewidth=2, color='black'),
        plt.Rectangle((-80, -47.5), 160, 190, linewidth=2, color='black', fill=False),
        plt.Circle((0, 142.5), radius=60, linewidth=2, color='black', fill=False),
        plt.Circle((0, 142.5), radius=60, linewidth=2, color='black', fill=False, linestyle='dashed'),
        plt.Arc((0, 0), 475, 475, theta1=0, theta2=180, linewidth=2, color='black'),
        plt.Rectangle((-250, -47.5), 500, 470, linewidth=2, color='black', fill=False),
    ]
    
    for element in court_elements:
        ax.add_patch(element)
        
def plot_shots_with_predictions(shots):
    plt.figure(figsize=(12, 11))
    ax = plt.gca()
    ax.set_facecolor('white')
    plot_court(ax)
    
    scatter = ax.scatter(shots['LOC_X'], shots['LOC_Y'], c=shots['PREDICTION_PROB'], 
                         cmap='RdYlBu', s=100, edgecolors='black')
    
    plt.colorbar(scatter, label='Predicted Make Probability')
    plt.title('In-Game Shot Predictions')
    plt.xlabel('Court X Position')
    plt.ylabel('Court Y Position')
    plt.show()
