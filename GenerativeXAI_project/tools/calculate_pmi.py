import os
import matplotlib.pyplot as plt


def compute_probs(x:str,y:str,responses:list[str]):
    x_count = 0
    y_count = 0
    joint_coint = 0
    total = len(responses)

    for pair in responses:
        x_present = x in pair
        y_present = y in pair
            
        if x_present and y_present:
            joint_coint += 1
        if x_present:
            x_count += 1
        if y_present:
            y_count += 1
    p_x = x_count/total if x_count else 1e-10
    p_y = y_count/total if y_count else 1e-10
    p_xy = joint_coint/total if joint_coint else 1e-10
    return p_x,p_y,p_xy

def visualize(words: list[str], anchor_word_idx: int, pmi_scores: list[float],
                  save_path: str | None = None) -> None:
    # Handle edge cases
    if len(pmi_scores) == 0:
        print("No PMI values to visualize")
        return
    
    min_pmi = min(pmi_scores)
    max_pmi = max(pmi_scores)
    
    normalized_scores = [(val - min_pmi) / (max_pmi - min_pmi) for val in pmi_scores]
    
    # Create figure
    _, ax = plt.subplots(figsize=(1.2*len(words), 1.0))
    ax.axis('off')
    
    # Color map: lighter for low PMI, darker for high PMI
    cmap = plt.get_cmap('YlOrRd')
    
    # Display words with background colors
    x_pos = 0.05
    y_pos = 0.6
    spacing = 0.99 / len(words)  # Distribute words across figure width
    
    for i, word in enumerate(words):

        if i == anchor_word_idx:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                            edgecolor='blue', linewidth=2)
        else:
            color = cmap(normalized_scores.pop(0))
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color, 
                            edgecolor='gray', linewidth=1)
            # Add PMI score below word
            ax.text(x_pos - 0.02, y_pos - 0.45, f'PMI: {pmi_scores.pop(0):.2f}', fontsize=9, 
               ha='left', va='center', style='italic', color='gray')
            
        ax.text(x_pos, y_pos, word, fontsize=14, ha='left', va='center',
                bbox=bbox_props)
        x_pos += spacing
    
    # plt.title(f'PMI Visualization (Anchor word: "{words[anchor_word_idx]}")', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()  
