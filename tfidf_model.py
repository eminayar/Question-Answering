import math
def computeTF(tokens):
    """
    Gets tokens of a document as a parameter
    Evaluates and returns the tf vector of this document
    """
    tf = {}
    terms = set(tokens)
    for term in terms:
        tf[term] = tokens.count(term)
    return tf

def computeDF(tokens, TERM_NO_DOCS):
    """
    Gets tokens of a document as a parameter
    Increments the counter of terms which will finally become the document frequency of the terms.
    """
    terms = set(tokens)
    for term in terms:
        if term in TERM_NO_DOCS:
            TERM_NO_DOCS[term] += 1
        else:
            TERM_NO_DOCS[term] = 1
    return TERM_NO_DOCS

def computeIDF(tokens, NO_DOCS, TERM_NO_DOCS):
    """
    Gets tokens of a document as a parameter
    Evaluates and returns the idf vector of this document
    """
    idf = {}
    terms = set(tokens)
    for term in terms:
        idf[term] = math.log(float(1 + NO_DOCS / (1 + TERM_NO_DOCS[term])))
    return idf

def normalize_vector(tf_idf):
    """
    Gets a tf_idf vector 
    Returns the length normalized version of it.
    """
    length = 0
    for term in tf_idf:
        length += tf_idf[term] ** 2
    length = math.sqrt(length)
    for term in tf_idf:
        tf_idf[term] = tf_idf[term] / length
    return tf_idf
def computeTF_IDF(tf, idf):
    """
    Gets the tf and idf vectors of a document,
    Returns the length normalized tf_idf vector of this document 
    """
    tf_idf={}
    for term in tf:
        tf_idf[term] = tf[term] * idf[term]
    
    tf_idf = normalize_vector(tf_idf)
    
    return tf_idf

def tf_idf_model(docs_by_tokens):
    """
    Get the movie_table
    Evaluate the tf-idf scores of each documents, then create tf-idf matrix(which is also stored as map)
    Then return this tf_idf matrix
    """
    NO_DOCS = len(docs_by_tokens)
    TERM_NO_DOCS = {}
    tf_idf_table = {}
    for document in docs_by_tokens:
        TERM_NO_DOCS = computeDF(document['tokens'], TERM_NO_DOCS)
        tf = computeTF(document['tokens'])
        idf = computeIDF(document['tokens'], NO_DOCS, TERM_NO_DOCS)
        tf_idf_table[document['id']] = computeTF_IDF(tf, idf)
    return tf_idf_table
    
