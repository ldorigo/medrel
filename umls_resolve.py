from utils import QUICKUMLS_LOCATION_WIDGET
import constants
from QuickUMLS.quickumls import QuickUMLS
import sqlite3 
import grammar_analysis


def get_cui_db():
    db = sqlite3.connect("data/cuis.db")
    print("Loaded database")
    return db
    
def get_qmatcher():
    return QuickUMLS(
    QUICKUMLS_LOCATION_WIDGET.value,
    accepted_semtypes=constants.ACCEPTED_CATEGORIES,
    overlapping_criteria='score')

def resolve_sign(sign,qmatcher):
    matches = qmatcher._get_matches_from_tokenlist(sign[0])
    res = []
    for m in matches:
        if m['similarity'] > 0.8:
            res.append(m['cui'])
    if res:
        return (res, sign[1])
    else:
        return None
    
def resolve_relations(rels, qmatcher):
    results = []
    for rel in rels:
        l = resolve_sign(rel[0],qmatcher)
        r = resolve_sign(rel[2],qmatcher)
        if l and r:
            results.append((l, rel[1], r))
    return results
        

def pretty_print_cui_relations(rels, db):
    c = db.cursor()
    for r in rels[2]:
        named_signs_l = []
        named_signs_r = []
        for cui in r[0][0]:
            c.execute("SELECT canonical_name FROM cuis WHERE cui=?", (cui,))
            named_signs_l.append(c.fetchone())
        for cui in r[2][0]:
                    c.execute("SELECT canonical_name FROM cuis WHERE cui =?", (cui,))
                    named_signs_r.append(c.fetchone())

        link = grammar_analysis.get_pretty_rel_indicator(r)
        print(
            "[{} ({})] {} [{} ({})]".format(
                named_signs_l, grammar_analysis.mod_to_symbol(r[0][1]), link, named_signs_r, grammar_analysis.mod_to_symbol(r[2][1])

            )
        )

def get_cui_relations_generator(relations_generator, qmatcher):
    for rel in relations_generator:
        res = resolve_relations(rel[2], qmatcher)
        if res:
            yield (rel[0],rel[1],res)
            

if __name__ == "__main__":
    # complex_sentence = "I eat both big and small young or old apples."
    complex_sentence = "By meta-analysis, we found A1166C polymorphism was associated with decreased risk for breast cancer in Caucasian population in an additive model . "

    
    nlp = grammar_analysis.get_nlp()
    resiter = grammar_analysis.get_relation_generator(nlp)
    qmatcher = get_qmatcher()
    gen = get_cui_relations_generator(resiter, qmatcher)
    t= next(gen)
    db = get_cui_db()
    pretty_print_cui_relations(t,db)

