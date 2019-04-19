from abbreviations import schwartz_hearst


def get_abbreviations(text):
    """
    Get list of abbreviations in sentence
    """
    pairs = schwartz_hearst.extract_abbreviation_definition_pairs(
        doc_text=text)
    return pairs


