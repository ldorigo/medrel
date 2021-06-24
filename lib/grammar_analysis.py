# import doctest
import inspect
import logging
from enum import Enum
from typing import Dict, Generator, Iterable, List, Optional, Tuple, cast
import spacy
from spacy import displacy
from spacy import tokens

# from spacy.attrs import ENT_IOB, ENT_TYPE
# from spacy.lang.en import LEMMA_EXC, LEMMA_INDEX, LEMMA_RULES, English
# from spacy.lemmatizer import Lemmatizer
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token

# from termcolor import colored
# from tqdm.auto import tqdm

import lib.constants as constants
from spacy.vocab import Vocab

# import QuickUMLS.toolbox as tb
# from QuickUMLS.quickumls import QuickUMLS

# from utils import QUICKUMLS_LOCATION_WIDGET

# lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)


class Modifier(Enum):
    NEUTRAL = 0
    POSITIVE = 1
    NEGATIVE = -1


############################
## Type definitions #########
############################

Demultiplication = List[List[Token]]
Meaning = Tuple[Demultiplication, Modifier]
Meanings = List[Meaning]
Relation = Tuple[Meaning, Modifier, Meaning]
Relations = List[Relation]


class NounToken(Token):
    pass


class AdjToken(Token):
    pass


############################
## Module initialization ###
############################


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.addLevelName(5, "TRACE")
logging.basicConfig(format="%(message)s")


def logwithdepth(message: str):
    if logger.getEffectiveLevel() == 4:
        depth = len(inspect.stack()) - 9
        logger.log(5, "→→" * depth + " " + message)


# def get_nlp(model_path: str = None) -> English:
#     if model_path is None:
#         model_path = "./data/annotate/scispacy_myner/model-best/"
#     n = spacy.load(model_path)
#     print("Loaded Model")
#     return n


############################
## Matchers ################
############################


def get_matchers(vocab: Vocab) -> Dict[str, Matcher]:
    matcher_relations = Matcher(vocab)
    matcher_negations = Matcher(vocab)
    matcher_relations.add(
        "neutral_relation_indicator", constants.PATTERNS_NEUTRAL_RELATIONS
    )
    matcher_negations.add("negation", constants.PATTERNS_NEGATION)

    return {"relations": matcher_relations, "negations": matcher_negations}


############################
## Utility functions #######
############################
def renderent(doc: Doc):
    displacy.render(doc, style="ent", jupyter=True)


def renderdep(doc: Doc):
    displacy.render(
        doc,
        style="dep",
        jupyter=True,
        options={"compact": True, "collapse_punct": False, "fine_grained": False},
    )


def mod_to_symbol(mod: Modifier):
    if mod == Modifier.NEUTRAL:
        return "~"
    elif mod == Modifier.NEGATIVE:
        return "↓"
    elif mod == Modifier.POSITIVE:
        return "↑"


def get_pretty_rel_indicator(r: Relation):
    if r[1] == Modifier.NEUTRAL:
        return "<->"
    elif r[1] == Modifier.NEGATIVE:
        return "<--"
    else:
        return "-->"


def pretty_print_relation(r: Relation) -> str:
    link = get_pretty_rel_indicator(r)
    result = "[{} ({})] {} [{} ({})]".format(
        r[0][0], mod_to_symbol(r[0][1]), link, r[2][0], mod_to_symbol(r[2][1])
    )
    return result


# def get_relation_generator(nlp: Language, path: str = None):
#     if path is None:
#         path = "./data/sessions/small_subset/relevant_sentences_plaintext.txt"
#     with open(path, "r") as f:
#         for sentjson in f:
#             obj = json.loads(sentjson)
#             sent = obj["sentence"]
#             pmid = obj["pmid"]
#             d = nlp(sent)
#             try:
#                 res = parse_sentence(d, nlp)
#             except:
#                 logging.error("encountered error when parsing:")
#                 logging.error("'{}'".format(d))
#                 res = []
#             if res:
#                 yield ((d, pmid, res))


def surround_ents(doc: Doc) -> str:
    """Surround entities in a doc with ** for lightweight display (without displacy)

    Args:
        doc (Doc): Spacy DOC containing named entities

    Returns:
        str: string of the doc with entities surrounded by **
    """
    res: List[str] = []
    for tok in doc:
        if tok.ent_iob_ == "B":
            res.append("**")
        elif tok.ent_iob_ == "O" and doc[tok.i - 1].ent_iob_ == "I":
            res.append("**")
        res.append(tok.text)

    return " ".join(res)


############################
## Grammatical Checks ######
############################


def has_modifier_adjective(noun: NounToken) -> bool:
    """Check wether the noun is modifier by a single amount/intensity modifier ("higher","increased",...)"""

    adjs = get_noun_adjectives_roots(noun)
    if len(adjs) == 1:
        if (
            not has_conjunctions(adjs[0])
            and adjs[0].lower_ in constants.MODIFIER_ADJECTIVES
        ):
            return True
    return False


def has_nmod(noun_token: NounToken) -> bool:
    """Check if the noun given as argument has a noun modifier subclause

    >>> doc = nlp("Within invasive disease, seed localization was associated with lower rates of margin positivity.")
    >>> has_nmod(doc[10])
    True
    >>> has_nmod(doc[2])
    False
    """
    assert (
        noun_token.pos_ in constants.NOUN_POS_TAGS
    ), "Error: trying to check noun modifier of something that isn't a noun"
    for tok in noun_token.doc:
        if tok.dep_ == "nmod" and tok.head == noun_token:
            return True
    return False


def has_conjunctions(tok: Token) -> bool:
    """Check if the token given as arguments has conjunctions.
    Note: this only returns true if the conjunctions are on a lower level in the parse tree.
    """
    if get_conjunctions(tok):
        return True
    return False


def has_adjectives(noun_token: NounToken) -> bool:
    """Check if the noun is modifier by one or more adjectives."""
    assert (
        noun_token.pos_ in constants.NOUN_POS_TAGS
    ), "Error: trying to check adjectives of something that isn't a noun"
    for tok in noun_token.doc[0 : noun_token.i]:
        if tok.dep_ == "amod" and tok.head == noun_token:
            return True
    return False


def is_compound_noun(noun_token: NounToken) -> bool:
    """Check if the token is the root of a compound noun."""
    assert (
        noun_token.pos_ in constants.NOUN_POS_TAGS
    ), "Error: trying to check if something that isn't a noun it a compound noun"
    for tok in noun_token.doc[0 : noun_token.i]:
        if tok.dep_ == "compound" and tok.head == noun_token:
            return True
    return False


############################
## Grammatical operations ##
############################

# (Small operations such as getting a verb's subject)


def get_modifier_adjective(noun: NounToken) -> Optional[Tuple[AdjToken, Modifier]]:
    """Get the adjective that modifies the quantity-indicating noun given as argument ("levels","amount",...), as well as its direction (increase, lower, or neutral )

    Args:
        noun (NounToken): A quantity-indicating noun

    Returns:
        Optional[Tuple[AdjToken, Modifier]]: Tuple of (modifying adjective, direction)
    """
    assert (
        noun.lower_ in constants.QTY_NOUNS
        or noun.lower_ in constants.QTY_NOUNS_EXPERIMENTAL
    ), "Error: Trying to get quantity modifier of a noun that is not a quantity indicator"
    logwithdepth("Getting modifier adjective of quantity noun '{}'.".format(noun))
    adjs = get_noun_adjectives_roots(noun)
    if len(adjs) == 1:
        if (
            not has_conjunctions(adjs[0])
            and adjs[0].lower_ in constants.MODIFIER_ADJECTIVES
        ):
            if adjs[0].lower_ in constants.DECREASE_ADJECTIVES:
                mod = Modifier.NEGATIVE
            else:
                mod = Modifier.POSITIVE
            logwithdepth(
                "Found modifier adjective: '{}' with level {}.".format(adjs[0], mod)
            )
            return (adjs[0], mod)
    return None


def get_conjunctions(token: Token) -> List[Token]:
    """Get the conjunctions of the given token that are lower than it in the dependency parse tree."""
    logwithdepth("Getting conjunctions for token: '{}'".format(token))
    res: List[Token] = []
    for t in token.doc:
        if token in t.conjuncts and t.head == token:
            res.append(t)
    logwithdepth("Found conjunctions: {}.".format(res))
    return res


def get_nmod_root(noun_token: NounToken) -> Optional[NounToken]:
    """Get the root of the subclause that modifies the noun given as argument.

    >>> doc = nlp("Within invasive disease, seed localization was associated with lower rates of margin positivity.")
    >>> get_nmod_root(doc[10])
    positivity
    >>> get_nmod_root(doc[2])
    """
    assert (
        noun_token.pos_ in constants.NOUN_POS_TAGS
    ), "Error: trying to get noun modifier of something that isn't a noun"

    for tok in noun_token.doc:
        if tok.dep_ == "nmod" and tok.head == noun_token:
            logwithdepth("Got nmod root of '{}': '{}'".format(noun_token, tok))
            return cast(NounToken, tok)
    logwithdepth("Found no root for nmod of {}.".format(noun_token))
    return None


def get_nmod_prep(noun_token: NounToken) -> Optional[Token]:
    """Get the preposition that links the noun given as argument to its noun modifier subclause

    >>> doc = nlp("Within invasive disease, seed localization was associated with lower rates of margin positivity.")
    >>> get_nmod_case(doc[10])
    of
    >>> get_nmod_root(doc[2])
    """
    r = get_nmod_root(noun_token)
    if r is None:
        raise ValueError(
            "Error: Trying to get the preposition to a non-existing subclause."
        )
    for tok in r.doc:
        if tok.dep_ == "case" and tok.head == r:
            return tok
    return None


def get_closest_adjective(noun_token: NounToken) -> Optional[AdjToken]:
    """Return the closest adjective of the given noun."""
    assert (
        noun_token.pos_ in constants.NOUN_POS_TAGS
    ), "Error: trying to get adjectives of something that isn't a noun"
    closest = None
    for tok in noun_token.doc[0 : noun_token.i]:
        if tok.dep_ == "amod" and tok.head == noun_token:
            closest = tok
    return cast(AdjToken, closest)


def get_noun_adjectives_roots(noun_token: NounToken) -> List[AdjToken]:
    """Get a list of the roots of the adjectival modifiers of the given noun."""
    logwithdepth("Getting all first-level adjectives of noun: '{}'.".format(noun_token))
    assert (
        noun_token.pos_ in constants.NOUN_POS_TAGS
    ), "Error: trying to get adjectives of something that isn't a noun"

    adjs: List[AdjToken] = []
    for tok in noun_token.doc[0 : noun_token.i]:
        if tok.dep_ == "amod" and tok.head == noun_token:
            adjs.append(cast(AdjToken, tok))
    logwithdepth("Found 1st-level adjectives: {}.".format(adjs))
    return adjs


def resolve_compound_noun(noun_token: NounToken) -> List[NounToken]:
    """Given a noun, return a list comprising the full compound noun corresponding to it.
    If the noun is not compound, the list only contains the noun itself.
    """
    logwithdepth("Resolving compound noun with root: '{}'".format(noun_token))
    assert (
        noun_token.pos_ in constants.NOUN_POS_TAGS
    ), "Error: trying to resolve a compound noun on something that isn't a noun"

    noun_group: List[NounToken] = []
    for tok in noun_token.doc[0 : noun_token.i]:
        if tok.dep_ == "compound" and tok.head == noun_token:
            noun_group.append(cast(NounToken, tok))
    noun_group.append(noun_token)
    logwithdepth("Resolved compound noun to: {}".format(noun_group))
    return noun_group


def resolve_adjective_conjunctions(adjective_token: AdjToken) -> List[Token]:
    logwithdepth("Resolving adjective conjunctions for: '{}'.".format(adjective_token))
    assert (
        adjective_token.pos_ == "ADJ" or adjective_token.dep_ == "amod"
    ), "Error: trying to resolve adjective conjunctions on something that isn't an adjective."
    res: List[Token] = [cast(Token, adjective_token)] + get_conjunctions(
        adjective_token
    )
    logwithdepth("Found conjunctions: {}.".format(res))
    return res


def demultiply_noun_adjectives(
    noun_token: NounToken, full_noun: List[NounToken]
) -> Demultiplication:
    """Return a list of all the distinct semantic meanings that can be derived from a noun and its adjectives.

    Example: "big and small young or old apples" ->
        [
            [apples],
            [young, apples], [old, apples], [big, apples], [small, apples],
            [big, young, apples], [big, old, apples], [small, young, apples], [small, old, apples]
        ]
    """
    logwithdepth(
        "Generating list of adjective sequences for noun: '{}' (full noun: {}).".format(
            noun_token, full_noun
        )
    )
    adjectives = get_noun_adjectives_roots(noun_token)
    full_adjectives = [resolve_adjective_conjunctions(a) for a in adjectives]
    reversed_full = full_adjectives[::-1]
    ## Example: [["yellow","green","red"], ["small","big"]]
    results: List[List[Token]] = [cast(List[Token], full_noun)]
    for adj_level in reversed_full:
        # Ex.: ["yellow","green","red"]
        temp = [[a] + cast(List[Token], r) for a in adj_level for r in results]
        results += temp

    results.sort(key=len, reverse=True)

    logwithdepth("Generated adjective sequences: {}".format(results))
    return results


def adjectives_to_modifiers(meanings: Meanings) -> Meanings:
    """Check the given list of meanings and replace inner adjectives that give a quantity information (higher, stronger, etc.) by outer modifiers"""
    logwithdepth("Checking adjective sequences for modifiers: {}".format(meanings))
    newmeanings: Meanings = []
    for meaning in meanings:
        assert meaning is not None
        if meaning[0][0].lower_ in constants.MODIFIER_ADJECTIVES:
            assert (
                meaning[1] == Modifier.NEUTRAL
            ), "Resolving adjectives of a non-neutral sign"

            if meaning[0][0].lower_ in constants.INCREASE_ADJECTIVES:
                newmeanings.append((meaning[0][1:], Modifier.POSITIVE))
                logwithdepth(
                    "Changed adjective '{}' to positive modifier".format(meaning[0][0])
                )
            elif meaning[0][0].lower_ in constants.DECREASE_ADJECTIVES:
                newmeanings.append((meaning[0][1:], Modifier.NEGATIVE))
                logwithdepth(
                    "Changed adjective '{}' to negative modifier".format(meaning[0][0])
                )
            elif meaning[0][0].lower_ in constants.NEUTRAL_ADJECTIVES:
                newmeanings.append((meaning[0][1:], Modifier.NEUTRAL))
                logwithdepth(
                    "Changed adjective '{}' to neutral modifier".format(meaning[0][0])
                )
            else:
                assert False, "ERROR: shouldn't happen"
        else:
            newmeanings.append(meaning)
    return newmeanings


def resolve_noun_modifier_clause(
    modified_noun: NounToken,
    adj_demultiplications: Demultiplication,
    external_modifier: Optional[Modifier],
) -> Meanings:
    """[summary]

    [extended_summary]

    Args:
        modified_noun (NounToken): [description]
        adj_demultiplications (Demultiplication): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    logwithdepth(
        "Resolving meanings of noun modifier clause that modify: '{}' and external modifier {}. Root has adjective sequences: {}. ".format(
            modified_noun, external_modifier, adj_demultiplications
        )
    )
    assert (
        modified_noun.pos_ in constants.NOUN_POS_TAGS
    ), "Error: Trying to resolve a noun clause with non-noun root"

    subclause_root = get_nmod_root(modified_noun)

    if subclause_root is None:
        meanings: Meanings = [(a, external_modifier) for a in adj_demultiplications]
        return adjectives_to_modifiers(meanings)
    if subclause_root.lower in constants.BREAKWORDS:
        logwithdepth(
            "Found a breakword ({}). Not going further.".format(subclause_root)
        )
        return [(a, external_modifier) for a in adj_demultiplications]

    # If the noun modified by the subclause is a noun expressing quandity:
    if (
        modified_noun.lower_ in constants.QTY_NOUNS
        or modified_noun.lower_ in constants.QTY_NOUNS_EXPERIMENTAL
    ):
        logwithdepth(
            "Detected subclause that is quantified by external uantity noun + adjective ({}).".format(
                modified_noun
            )
        )
        if has_modifier_adjective(modified_noun):
            res = get_modifier_adjective(modified_noun)
            assert res, "Shouldn't happen!!"
            ad, mod = res

            logwithdepth(" ... which is modified by the adjective '{}'".format(ad))

            # This should only happen without external modifiers...
            assert external_modifier == Modifier.NEUTRAL
            final = parse_noun_clause(subclause_root, mod)
        else:
            logwithdepth("Got no modifier adjective.")
            final = parse_noun_clause(subclause_root, external_modifier)
    # elif modified_noun.lower_ in constants.QTY_NOUNS_EXPERIMENTAL:
    # logwithdepth("Found subclause of experimental quantity noun: '{}'".format(modified_noun))
    # final = parse_noun_clause(subclause_root, external_modifier)
    # If the noun itself is a quantifier ("increase", "decreases")
    elif modified_noun.lower_ in constants.QUANTIFIER_NOUNS:
        logwithdepth(
            "Detected subclause that is quantified by external noun ({}).".format(
                modified_noun
            )
        )
        # The modified noun is of the type "an increase", "an increment".
        # The actual interesting information is in the subclause, the upper-level noun is actually the modifier.
        if modified_noun.lower_ in constants.INCREASE_NOUNS:
            modifier = Modifier.POSITIVE
        elif modified_noun.lower_ in constants.NEUTRAL_NOUNS:
            modifier = Modifier.NEUTRAL
        else:
            modifier = Modifier.NEGATIVE
        # This should only happen without external modifiers...
        assert external_modifier == Modifier.NEUTRAL
        final = parse_noun_clause(subclause_root, modifier)
    else:
        # The subclause acts as a modifier, and we demultiply it with any existing demultiplications
        logwithdepth(
            "Clause with root {} modifies noun {}.".format(
                subclause_root, modified_noun
            )
        )
        logwithdepth("Getting meanings of the clause.")
        submeanings = parse_noun_clause(subclause_root)
        logwithdepth("Got meanings: {}.".format(submeanings))
        # if submeanings is None:
        #     raise ValueError
        subclauses = [el[0] for el in submeanings if el is not None]
        adp = get_nmod_prep(modified_noun)
        if adp is None:
            raise ValueError("Error: nmod clause has no adposition")
        results = adj_demultiplications.copy()
        temp = []
        for sc in subclauses:
            temp += [r + [adp] + sc for r in results]

        results += temp

        result_meanings: Meanings = []
        for r in results:
            result_meanings.append((r, external_modifier))
        final = adjectives_to_modifiers(result_meanings)
    logwithdepth("Meanings extracted from clause: {}.".format(final))
    return final


def parse_noun_clause(
    root: NounToken, external_modifier: Optional[Modifier] = Modifier.NEUTRAL
) -> Meanings:
    """Parse a clause whose root is a noun (usually a sign as recognized by the statistical model),
    and return all meanings (=signs) that were found in it.

    Args:
        root (NounToken): Root of the noun clause
        external_modifier (Modifier, optional): If the clause's meaning was modified by an external modifier. Defaults to Modifier.NEUTRAL.

    Returns:
        Meanings: list of tuples of (sign,modifier)
    """

    logwithdepth(
        "Parsing noun clause with root: '{}' and external modifier {}.".format(
            root, external_modifier
        )
    )
    # assert (
    #     root.pos_ in constants.NOUN_POS_TAGS
    # ), "Error: Trying to parse a noun subsentence with non-noun root: {}".format(root)

    if root.lower_ in constants.BREAKWORDS:
        logwithdepth("Root of noun clause is a break work. Not going further.")
        return []
    full_root: List[NounToken]
    # Compound nouns should form a coherent unit, so they can be considered on their own:
    if is_compound_noun(root):
        full_root = resolve_compound_noun(root)
    else:
        full_root = [root]

    results: Meanings
    results = []

    # We get the list of all subslists of adjectives that modify the root
    adj_demultiplications = demultiply_noun_adjectives(root, full_root)

    results = resolve_noun_modifier_clause(
        root, adj_demultiplications, external_modifier
    )

    if has_conjunctions(root):
        # If the root of the noun (sub-)sentence has conjunctions, return each of them as a separate
        # return [parse_noun_subsentence(c,external_modifier) for c in root.conjuncts]
        # logger.warn(
        # "Warning: The root of the noun sentence ({}) has a conjunction that was ignored.".format(root))
        conjs = get_conjunctions(root)
        logwithdepth(
            "Root of noun clause ({}) has conjunctions ({}). Parsing them.".format(
                root, conjs
            )
        )
        for c in conjs:
            if c.dep_ in constants.NOUN_POS_TAGS:
                results += parse_noun_clause(c)
    logwithdepth("Extracted meanings from noun clause: {}.".format(results))
    return results
    # return [(adj_demultiplications, external_modifier)]


def get_relation_indicators(doc: Doc) -> List[Span]:
    """Return any relation indicators in the doc.

    Args:
        doc (Doc): [description]

    Returns:
        List[Span]: list of Spans containing relation indicators
    """
    logwithdepth("Getting relation indicators of sentence.")
    matchers = get_matchers(doc.vocab)
    relation_matches = matchers["relations"](doc)
    relation_markers = [doc[i:j] for _, i, j in relation_matches]
    logwithdepth("Found relation markers: {}.".format(relation_markers))
    return relation_markers


def is_parsable_sentence(doc: Doc) -> bool:
    """Check wether the sentence in the doc has a form that is currently parseable by this library.

    Args:
        doc (Doc): [description]

    Returns:
        bool: wether the sentence can currently be parsed
    """
    logwithdepth("Checking if sentence '{}' is parsable.".format(doc))
    matchers = get_matchers(doc.vocab)
    if matchers["negations"](doc):
        logger.debug(
            "Discarded sentence because it contains negation(s): \n {}".format(doc)
        )
        return False

    relation_indicators = get_relation_indicators(doc)

    if len(relation_indicators) > 1:
        logger.debug(
            "Discarded sentence because it contains more than one relation: \n {}".format(
                doc
            )
        )
        return False
    elif len(relation_indicators) == 0:
        logger.debug(
            "Discarded sentence because it contains no relations: \n {}".format(doc)
        )
        return False
    # elif len(doc.ents) != 2:
    #     logger.debug(
    #         "Discarded sentence because it contains more or less than two entities: "
    #     )
    #     logger.debug(surround_ents(doc))
    #     return False
    logwithdepth("Sentence was deemed parsable.")
    return True


def filter_duplicate_relations(rels: Relations) -> Relations:
    logwithdepth("Filtering relations:")

    for i, rel in enumerate(rels):
        logwithdepth("  {}. {} ".format(i, rel))
    unique_ids = set(range(0, len(rels)))
    unique_rels: Relations = []

    for i, rel in enumerate(rels):
        j = i + 1
        for other_rel in rels[i + 1 :]:
            if rel == other_rel:
                unique_ids.remove(i)
            elif rel[0] == other_rel[0] and rel[1] == other_rel[1]:
                # First sign and relation type are identical
                if rel[2][0] == other_rel[2][0]:
                    # second sign is also equal
                    if (
                        rel[2][1] in [Modifier.POSITIVE, Modifier.NEGATIVE]
                        and other_rel[2][1] == Modifier.NEUTRAL
                    ):
                        try:
                            unique_ids.remove(j)
                        except KeyError:
                            pass
                    elif (
                        other_rel[2][1] in [Modifier.POSITIVE, Modifier.NEGATIVE]
                        and rel[2][1] == Modifier.NEUTRAL
                    ):
                        try:
                            unique_ids.remove(i)
                        except KeyError:
                            pass
                    else:
                        assert (
                            False
                        ), "Error: relations are neither equal nor different??"
            elif rel[2] == other_rel[2] and rel[1] == other_rel[1]:
                # Second sign and relation type are identical
                if rel[0][0] == other_rel[0][0]:
                    # second sign is also equal
                    if (
                        rel[0][1] in [Modifier.POSITIVE, Modifier.NEGATIVE]
                        and other_rel[0][1] == Modifier.NEUTRAL
                    ):
                        try:
                            unique_ids.remove(j)
                        except KeyError:
                            pass
                    elif (
                        other_rel[0][1] in [Modifier.POSITIVE, Modifier.NEGATIVE]
                        and rel[0][1] == Modifier.NEUTRAL
                    ):
                        try:
                            unique_ids.remove(i)
                        except KeyError:
                            pass
                    else:
                        assert (
                            False
                        ), "Error: relations are neither equal nor different??"
            j += 1

    logwithdepth("Kept relations {}".format(unique_ids))

    for i in unique_ids:
        unique_rels.append(rels[i])
    return unique_rels


def most_precise_relations(rels: Relations) -> Relations:
    res = [rels[0]]
    # Length of the signs on both sides of the relation:
    l1_max = len(res[0][0][0])
    l2_max = len(res[0][2][0])

    for rel in rels[1:]:
        l1 = len(rel[0][0])
        l2 = len(rel[2][0])
        if (l1 > l1_max and l2 >= l2_max) or (l1 >= l1_max and l2 > l2_max):
            res = [rel]
        elif l1 == l1_max and l2 == l2_max:
            res.append(rel)

    return res


def parse_sentence(doc: Doc) -> Relations:
    """Parse a full sentence and return a list of all relations

    Args:
        doc (Doc): Spacy doc containing the (spacy-)parsed sentence
        nlp ([type]): spacy Lang object

    Returns:
        [type]: [description]
    """
    logwithdepth("Attempting to parse sentence: '{}'.".format(doc))
    if not is_parsable_sentence(doc):
        return []

    relation_indicator = get_relation_indicators(doc)[0]

    # left_side: Span = doc[0:relation_indicator.start]
    # right_side: Span = doc[relation_indicator.end:-1]

    # try:
    #     left_side = [e for e in doc.ents if e.end <= relation_indicator.start][0]
    #     right_side = [e for e in doc.ents if e.start >= relation_indicator.end][0]
    # except IndexError:
    #     return []

    rel_indicator_verb = [w for w in relation_indicator if w.lemma_ == "associate"][0]
    # left_root = left_side.root
    try:
        left_root = [
            tok
            for tok in rel_indicator_verb.children
            if tok.dep_ in ["nsubjpass", "subj"]
        ][0]
    except IndexError:
        return []
    right_root = list(rel_indicator_verb.rights)[0]
    if (
        not right_root.pos_ in constants.NOUN_POS_TAGS
        or not left_root.pos_ in constants.NOUN_POS_TAGS
    ):
        return []
    # right_root = right_side.root

    l_ents = parse_noun_clause(left_root)
    r_ents = parse_noun_clause(right_root)

    relations: Relations = []
    for l_ent in l_ents:
        for r_ent in r_ents:
            relations.append((l_ent, Modifier.NEUTRAL, r_ent))

    ## We remove relations that are identical to each other except for a sign modifier.
    relations_unique = filter_duplicate_relations(relations)
    return relations_unique


def get_relations_generator(
    doc_and_sentence_generator: Iterable[Tuple[tokens.Doc, List[int]]]
) -> Generator[Tuple[tokens.Doc, Dict[int, Relations]], None, None]:
    for doc, sentence_indices in doc_and_sentence_generator:
        sents = list(doc.sents)
        results_dict: Dict[int, Relations] = {}
        for sentence_index in sentence_indices:
            sentence = sents[sentence_index]
            relations = parse_sentence(sentence.as_doc())
            if relations:
                results_dict[sentence_index] = relations
        yield (doc, results_dict)


if __name__ == "__main__":
    # complex_sentence = "I eat both big and small young or old apples."
    # complex_sentence = "By meta-analysis, we found A1166C polymorphism was associated with decreased risk for breast cancer in Caucasian population in an additive model . "
    complex_sentence = "By meta-analysis, we found A1166C polymorphism was associated with decreased risk for breast cancer in Caucasian population in an additive model ."

    logger.setLevel(4)

    # resiter = get_relation_generator(nlp)
    # next(resiter)

    # # demultiply_noun_adjectives(ns[-2], [ns[-2]])
    nlp = spacy.load("en_core_sci_md")
    ns = nlp(complex_sentence)
    p = parse_sentence(ns)
    for r in p:
        print(pretty_print_relation(r))
