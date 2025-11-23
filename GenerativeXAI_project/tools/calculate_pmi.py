import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_probs_new(x:str,y:str,responses:list[str]):
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

def visualize_new(words: list[str], anchor_word_idx: int, pmi_scores: list[float],
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

def compute_probabilities(sentence: str, responses: list[list[str]],
                          anchor_word_idx: int, prompts_per_word: int
                          ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Compute P(x), P(y), and P(x,y) for PMI calculation.
    
    Args:
        sentence: Input sentence
        responses: List of responses for each masked word pair
        anchor_word_idx: Index of the anchor word (x)
        prompts_per_word: Number of prompts generated per word
        
    Returns:
        Tuple of (p_xy, p_x, p_y) dictionaries
    """
    p_xy = defaultdict(float)
    p_x = defaultdict(float)
    p_y = defaultdict(float)
    
    words = sentence.lower().split()
    x = words[anchor_word_idx]
    
    # Build index mapping: skip anchor word when counting
    word_indices = [i for i in range(len(words)) if i != anchor_word_idx]
    
    for response_idx, response_list in enumerate(responses):
        # Get the corresponding word index for this response
        if response_idx >= len(word_indices):
            break
        y_idx = word_indices[response_idx]
        y = words[y_idx]
        
        # Each response_list contains multiple replacement pairs
        for pair in response_list:
            # Check if words appear in the generated replacement
            x_present = x in pair
            y_present = y in pair
            
            if x_present and y_present:
                p_xy[(x, y)] += 1
            if x_present:
                p_x[x] += 1
            if y_present:
                p_y[y] += 1
    
    # Normalize by total number of samples per word pair
    total_samples = prompts_per_word
    for key in p_xy:
        p_xy[key] /= total_samples
    for key in p_x:
        p_x[key] /= (total_samples * (len(words) - 1))  # x appears in all word pairs
    for key in p_y:
        p_y[key] /= total_samples # missing logic if a word appears multiple times
    
    return p_xy, p_x, p_y


def calculate_pmi(sentence: str, anchor_word_idx: int, p_xy: dict, 
                  p_x: dict, p_y: dict) -> dict[str, float]:
    """
    Calculate PMI for each word pair.
    
    Args:
        sentence: Input sentence
        anchor_word_idx: Index of the anchor word
        p_xy: Joint probability dictionary
        p_x: Marginal probability for x
        p_y: Marginal probability for y
        
    Returns:
        Dictionary mapping words to their PMI values with the anchor word
    """
    words = sentence.lower().split()
    x = words[anchor_word_idx]
    pmi_scores = {}
    
    for i, y in enumerate(words):
        if i == anchor_word_idx:
            pmi_scores[y] = 0.0  # PMI with itself
            continue
            
        # Get probabilities
        prob_xy = p_xy.get((x, y), 1e-10)  # Small value to avoid log(0)
        prob_x = p_x.get(x, 1e-10)
        prob_y = p_y.get(y, 1e-10)
        
        # Calculate PMI
        if prob_xy > 0 and prob_x > 0 and prob_y > 0:
            pmi = np.log2(prob_xy / (prob_x * prob_y))
        else:
            pmi = 0.0
            
        pmi_scores[y] = pmi
    
    return pmi_scores


def visualize_pmi(sentence: str, anchor_word_idx: int, pmi_scores: dict[str, float],
                  save_path: str | None = None) -> None:
    """
    Visualize PMI scores as a colored text display.
    
    Args:
        sentence: Input sentence
        anchor_word_idx: Index of the anchor word
        pmi_scores: Dictionary mapping words to PMI scores
        save_path: Optional path to save the figure
    """
    words = sentence.lower().split()
    
    # Normalize PMI scores to [0, 1] for coloring
    pmi_values = [pmi_scores.get(word, 0.0) for word in words]
    
    # Handle edge cases
    if len(pmi_values) == 0:
        print("No PMI values to visualize")
        return
    
    min_pmi = min(pmi_values)
    max_pmi = max(pmi_values)
    
    if max_pmi - min_pmi > 0:
        normalized_scores = [(val - min_pmi) / (max_pmi - min_pmi) for val in pmi_values]
    else:
        normalized_scores = [0.5] * len(pmi_values)
    
    # Create figure
    _, ax = plt.subplots(figsize=(1.2*len(words), 1.0))
    ax.axis('off')
    
    # Color map: lighter for low PMI, darker for high PMI
    cmap = plt.get_cmap('YlOrRd')
    
    # Display words with background colors
    x_pos = 0.05
    y_pos = 0.6
    spacing = 0.99 / len(words)  # Distribute words across figure width
    
    for i, (word, score, norm_score) in enumerate(zip(words, pmi_values, normalized_scores)):
        # Use bounding box with color
        color = cmap(norm_score)
        
        # Highlight the anchor word differently
        if i == anchor_word_idx:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                            edgecolor='blue', linewidth=2)
        else:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color, 
                            edgecolor='gray', linewidth=1)
        
        ax.text(x_pos, y_pos, word, fontsize=14, ha='left', va='center',
                bbox=bbox_props)
        
        # Add PMI score below word
        ax.text(x_pos - 0.02, y_pos - 0.45, f'PMI: {score:.2f}', fontsize=9, 
               ha='left', va='center', style='italic', color='gray')

        x_pos += spacing
    
    # plt.title(f'PMI Visualization (Anchor word: "{words[anchor_word_idx]}")', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def print_scores(p_xy: dict, p_x: dict, p_y: dict, pmi_scores: dict,
                 sentence: str, anchor_word_idx: int) -> None:
    """
    Print computed probabilities in a readable format.
    
    Args:
        p_xy: Joint probability dictionary
        p_x: Marginal probability for x
        p_y: Marginal probability for y
    """
    print("COMPUTED PROBABILITIES:")    
    print("P(x) - Marginal probability of anchor word:")
    for word, prob in sorted(p_x.items(), key=lambda x: x[1], reverse=True):
        print(f"  P({word}) = {prob:.4f}")
    
    print("P(y) - Marginal probability of other words:")
    for word, prob in sorted(p_y.items(), key=lambda x: x[1], reverse=True):
        print(f"  P({word}) = {prob:.4f}")
    
    print("P(x,y) - Joint probability:")
    for (x, y), prob in sorted(p_xy.items(), key=lambda x: x[1], reverse=True):
        print(f"  P({x}, {y}) = {prob:.4f}")
    
    print("PMI Scores:")
    for word, score in pmi_scores.items():
        print(f"  PMI({sentence.split()[anchor_word_idx].lower()}, {word}) = {score:.4f}")