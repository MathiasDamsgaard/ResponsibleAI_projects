import string


def is_response_valid(original_words, replaced_words):
    # Extract non-[MASK] words from the original sentence
    non_mask_words = [word for word in original_words if "[mask]" not in word]

    # Check if both original_words and replaced_words contain words
    if not non_mask_words or not replaced_words:
        return False

    # Check if each non-[MASK] word is present in the replaced sentence
    for word in non_mask_words:
        if word not in replaced_words:
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
                while replaced_word != stop_word:
                    replacements[replacement_idx].append(replaced_word)
                    replaced_word = replaced_words.pop(0)

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
    replacements = [" ".join(replacement) for replacement in replacements]
    return replacements

