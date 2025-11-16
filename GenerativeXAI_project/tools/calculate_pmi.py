# PMI(x,y)=log_2⁡ \frac{p(\{x,y\}| s-\{x,y\})}{P(\{x\}|s-\{x,y\})P(\{y\}|s-\{x,y\})}
# * Compute the $P(x)$, $P(y)$ and $P(x,y)$ first and print it out.
# * Compute the PMI for each word.
# * Visualize the result by coloring. Tips: you might need to normalize the result first. 

import numpy as np
from collections import defaultdict

def compute_probabilities(sentence: str, responses: list[list[str]],
                          anchor_word_idx: int , prompts_per_word: int
                          ) -> dict[str, float]:
    p_xy = defaultdict(float)
    p_x = defaultdict(float)
    p_y = defaultdict(float)
    x = sentence.lower().split()[anchor_word_idx]

    for i, response in enumerate(responses):
        y = sentence.lower().split()[i+anchor_word_idx+1] # only works for anchor_word_idx = 0

        for pair in response:
            if x in pair and y in pair:
                p_xy[f"p({x},{y})"] += 1 / prompts_per_word
            if x in pair:
                p_x[f"p({x})"] += 1 / prompts_per_word
            if y in pair:
                p_y[f"p({y})"] += 1 / prompts_per_word
    
    return p_xy, p_x, p_y


def calculate_pmi(p_xy, p_x, p_y):
    p_xy, p_x, p_y = map(lambda x: np.array(list(x.values())), [p_xy, p_x, p_y])

    if p_x.any() == 0 or p_y.any() == 0:
        return 0  # To avoid division by zero
    pmi = np.log2(p_xy / (p_x * p_y))
    return pmi