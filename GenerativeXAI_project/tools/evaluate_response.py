import string


def is_response_valid(original_words, replaced_words):
    # Enforce strict ordering and single-token replacement for each [mask].
    # original_words and replaced_words are lists of tokens (lowercase, punctuation removed).
    mask_count = original_words.count("[mask]")
    
    # Build list of (word, original_position) for non-mask words
    non_mask_items = [(w, i) for i, w in enumerate(original_words) if w != "[mask]"]

    # If there are no non-mask words, replaced_words must contain exactly one token per mask
    if not non_mask_items:
        return len(replaced_words) == mask_count

    if not replaced_words:
        return False

    # Find positions of non-mask words in replaced_words in order
    replaced_positions = []
    start_idx = 0
    for word, _ in non_mask_items:
        try:
            pos = replaced_words.index(word, start_idx)
        except ValueError:
            return False
        replaced_positions.append(pos)
        start_idx = pos + 1

    # Now verify the number of tokens between matched non-mask words equals the number of masks
    # in the corresponding segment of the original sentence. Each mask must correspond to exactly
    # one token in the generated sentence.
    # Build list of counts of masks in each segment of the original sentence.
    seg_mask_counts = []
    
    # masks before first non-mask
    first_orig_idx = non_mask_items[0][1]
    seg_mask_counts.append(original_words[:first_orig_idx].count("[mask]"))

    # masks between consecutive non-mask words
    for i in range(len(non_mask_items) - 1):
        left_orig_idx = non_mask_items[i][1]
        right_orig_idx = non_mask_items[i + 1][1]
        # count masks strictly between left and right
        seg_mask_counts.append(original_words[left_orig_idx + 1:right_orig_idx].count("[mask]"))

    # masks after last non-mask
    last_orig_idx = non_mask_items[-1][1]
    seg_mask_counts.append(original_words[last_orig_idx + 1:].count("[mask]"))

    # Now get actual token counts in replaced_words segments
    actual_counts = []
    # before first non-mask
    actual_counts.append(replaced_positions[0])
    # between matched non-masks
    for i in range(len(replaced_positions) - 1):
        actual_counts.append(replaced_positions[i + 1] - replaced_positions[i] - 1)
    # after last non-mask
    actual_counts.append(len(replaced_words) - replaced_positions[-1] - 1)

    # Compare expected and actual counts; each mask must map to exactly one token
    if len(seg_mask_counts) != len(actual_counts):
        return False

    for expected_masks, actual_tokens in zip(seg_mask_counts, actual_counts):
        if expected_masks != actual_tokens:
            return False

    return True


def get_stop_word(remaining_words):
    # If there are no more words, return False
    if len(remaining_words) > 1:
        # If the next word is a [MASK], return False
        if remaining_words[1] == "[mask]":
            return False
        # Else if the next word is a non-[MASK], return it
        else:
            return remaining_words[1]

    else:
        return False


def get_replacements(original_sentence, replaced_sentence):
    # remove punctuation and make lowercase
    custom_punctuation = string.punctuation.replace("[", "").replace("]", "")
    translation_table = str.maketrans("", "", custom_punctuation)
    original_sentence = original_sentence.translate(translation_table).lower()
    replaced_sentence = replaced_sentence.translate(translation_table).lower()
    # Split sentences into words
    original_words = original_sentence.split(" ")
    replaced_words = replaced_sentence.split(" ")
    # Find the words that replace [MASK]
    mask_count = original_words.count("[mask]")
    replacements = [[] for _ in range(mask_count)]
    replacement_idx = 0
    if not is_response_valid(original_words, replaced_words):
        print(f" Response is not valid. {original_words} {replaced_words}")
        return [""] * mask_count

    for check_idx, check_word in enumerate(original_words):
        if not replaced_words or len(replaced_words) == 0:
            # Handle the case when the list is empty
            continue
        replaced_word = replaced_words.pop(0)
        if check_word in ["[mask]", "[mask]."]:
            stop_word = get_stop_word(original_words[check_idx:])
            if stop_word:
                # search replaced_words until stop_word is found
                # guard against running out of words in replaced_words
                while replaced_word != stop_word:
                    replacements[replacement_idx].append(replaced_word)
                    if not replaced_words:
                        # stop_word not found in the remainder of replaced_words;
                        # break to avoid IndexError and continue processing
                        break
                    replaced_word = replaced_words.pop(0)

                # only push the stop_word back if we actually found it
                if replaced_word == stop_word:
                    replaced_words.insert(0, replaced_word)
                replacement_idx += 1
            else:
                if len(original_words[check_idx:]) > 1:
                    replacements[replacement_idx].append(replaced_word)
                    replacement_idx += 1
                else:
                    replaced_words.insert(0, replaced_word)
                    replacements[replacement_idx] = replaced_words
                    replacement_idx += 1

    # join words into sentences
    # Each replacement must be a single token; if multi-token replacement detected, treat as invalid
    for rep in replacements:
        if len(rep) > 1:
            print(f" Response contains multi-word replacement for a mask: {replacements}")
            return [""] * mask_count

    replacements = [" ".join(replacement) for replacement in replacements]
    return replacements

