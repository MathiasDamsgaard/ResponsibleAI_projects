def get_prompt(word_list, anchor_idx, other_idx):
    # Generate a single command
    new_word_list = word_list.copy()
    new_word_list[anchor_idx] = "[MASK]"
    new_word_list[other_idx] = "[MASK]"
    return " ".join(new_word_list)


def generate_prompts(sentence, anchor_idx):
    # Generate all commands for base_idx
    # sentence is a string of words
    # base_idx is the index of the word to be replaced with [MASK]
    word_list = sentence.split(" ")
    commands = []
    for other_idx in range(len(word_list)):
        if other_idx == anchor_idx:
            continue
        commands.append(get_prompt(word_list, anchor_idx, other_idx))
    return commands


def prefix_prompt(prompt):
    return f"Replace all instances of [MASK], in the following sentence, with one word each that make sense: {prompt}"
