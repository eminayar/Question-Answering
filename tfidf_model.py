import math
def computeTF(tokens):
    """
    Gets tokens of a document as a parameter
    Evaluates and returns the tf vector of this document
    """
    tf = {}
    terms = set(tokens)
    for term in terms:
        tf[term] = 1 + math.log10(tokens.count(term))
    return tf

def computeIDF(documents):
    """
    Gets tokens of a document as a parameter
    Increments the counter of terms which will finally become the document frequency of the terms.
    """
    idf = {}
    for document in documents:
        terms_in_this_doc = set(list(document['tokens']))
        for term in terms_in_this_doc:
            if term not in idf:
                idf[term] = 0
            idf[term] += 1
    
    for term in idf:
        idf[term] = math.log10(len(documents)/idf[term])
    
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
    idf = computeIDF(docs_by_tokens)
    tf_idf_table = {}
    for document in docs_by_tokens:
        tf = computeTF(document['tokens'])
        tf_idf_table[document['id']] = computeTF_IDF(tf, idf)
    return tf_idf_table, idf

def cos_similarity( paragraph1, paragraph2 ):
    cos_value = 0
    for term in paragraph1:
        cos_value += paragraph1[term] * paragraph2.get(term, 0)
    return cos_value 
    
