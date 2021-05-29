
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pyclustertend import hopkins
from sklearn.preprocessing import scale

def get_rocchio_query(baseline, top_docs, expansion_term_num):
    top_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    top_tfidf = top_tfidf_vectorizer.fit_transform(top_docs)
    feature_array = np.array(top_tfidf_vectorizer.get_feature_names())                
    tfidf_sorting = np.argsort(top_tfidf.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:expansion_term_num]
    extend_query = " "
    for key in top_n:
        extend_query = extend_query  + " "+ key
    return extend_query

def get_scp_query(baseline, top_docs, expansion_term_num):
    q_terms = set(list(baseline.split(" ")))
    scp_dict = dict()                                
    for q_term in q_terms:
        for top_doc in top_docs:
            doc_tokens = top_doc.split(" ")
            if q_term not in doc_tokens:
                continue
            q_index_list = [i for i, value in enumerate(doc_tokens) if value == q_term]
            for q_index in q_index_list:
                for d_index in range(len(doc_tokens)):
                    if abs(q_index-d_index) > 8:
                        continue
                    doc_token = doc_tokens[d_index]
                    if doc_token not in scp_dict.keys():
                        scp_dict[doc_token] = 1
                    else:
                        scp_dict[doc_token] = scp_dict[doc_token] + 1
                                      
    extend_query = " "
    sim_scores_sort = sorted(scp_dict.items(), reverse=True, key=lambda item: item[1])
    extend_term_num = 0 
    for key, _ in sim_scores_sort:
        extend_query = extend_query  + " "+ key
        extend_term_num += 1
        if extend_term_num == expansion_term_num:
            break
    return extend_query

def get_we_query(query, model, qe_type, expansion_term_num_list, max_extended_num):
    q_terms = set(query.split(" "))

    vector_list = []
    term_list = []
    for q_term in q_terms:
        if q_term not in model.wv.vocab:
            continue
        vector = model.wv.get_vector(q_term)
        vector_list.append(vector)
        term_list.append(q_term)

    mean_vector = np.mean(np.array(vector_list), axis=0)
    sim_terms = ()
    try:
        sim_terms = model.wv.similar_by_vector(mean_vector, topn = max_extended_num)
    except TypeError:
        sim_terms = ()
    
    extend_term_num = 0 
    extend_terms = []
    for key, _ in sim_terms:
        extend_terms.append(key)
        extend_term_num += 1
        if extend_term_num == max_extended_num:
            break
    
    query_list = []    
    extend_query = ""
    extend_num = 0
    for term in extend_terms:
        extend_query = extend_query + " "+ term
        extend_num += 1
        if extend_num in expansion_term_num_list:
            query_list.append(extend_query)
        
    return query_list


def get_we_query_static(query, model, qe_type, expansion_term_num):
    q_terms = set(query.split(" "))

    vector_list = []
    term_list = []
    for q_term in q_terms:
        if q_term not in model.wv.vocab:
            continue
        vector = model.wv.get_vector(q_term)
        vector_list.append(vector)
        term_list.append(q_term)

    extended_terms_list = []
    if qe_type =="cent":
        mean_vector = np.mean(np.array(vector_list), axis=0)
        sim_terms = ()
        try:
            sim_terms = model.wv.similar_by_vector(mean_vector, topn = 100)
        except TypeError:
            sim_terms = ()
        extend_term_num = 0
        extended_terms = ""
        for key, _ in sim_terms:
            extended_terms = extended_terms  + " "+ key 
            extend_term_num += 1
            if extend_term_num == expansion_term_num:
                break
        extended_terms_list.append(extended_terms)
    elif qe_type =="sum" or qe_type =="max":
        term_scores = {}
        for term in term_list:
            sim_terms = model.wv.similar_by_word(term, topn = 100)
            for sim_term, rel_score in sim_terms:
                # rel_score = model.wv.relative_cosine_similarity(term, sim_term, topn = expansion_threshold)
                if sim_term in term_scores:
                    if qe_type =="sum":                        
                        term_scores[sim_term] =  term_scores[sim_term] + rel_score
                    elif qe_type =="max":
                        if term_scores[sim_term] < rel_score:
                            term_scores[sim_term]  = rel_score
                else:
                    term_scores[sim_term] = rel_score
        
        term_scores_sort = sorted(term_scores.items(), reverse=True, key=lambda item: item[1])
        extend_term_num = 0
        extended_terms = ""
        for key, _ in term_scores_sort:
            extended_terms = extended_terms  + " "+ key 
            extend_term_num += 1
            if extend_term_num == expansion_term_num:
                break
        extended_terms_list.append(extended_terms)
    return extended_terms_list

def get_we_query_v2(summary, query, model, qe_type, expansion_threshold=50):
    sum_terms = set(summary.split(" "))
    sum_vector_list = []
    sum_term_list = []
    for s_term in sum_terms:
        if s_term not in model.wv.vocab:
            continue
        vector = model.wv.get_vector(s_term)
        sum_vector_list.append(vector)
        sum_term_list.append(s_term)
        
    sim_terms = ()
    if qe_type =="cent":
        mean_vector = np.mean(np.array(sum_vector_list), axis=0)
        try:
            sim_terms = model.wv.similar_by_vector(mean_vector, topn = expansion_threshold)
        except TypeError:
            sim_terms = ()
    elif qe_type =="max" or qe_type =="sum":        
        term_scores = {}
        for term in sum_term_list:
            sim_terms = model.wv.similar_by_word(term, topn = 100)
            for sim_term, rel_score in sim_terms:
                # rel_score = model.wv.relative_cosine_similarity(term, sim_term, topn = expansion_threshold)
                if sim_term in term_scores:
                    if qe_type =="sum":                        
                        term_scores[sim_term] =  term_scores[sim_term] + rel_score
                    elif qe_type =="max":
                        if term_scores[sim_term] < rel_score:
                            term_scores[sim_term]  = rel_score
                else:
                    term_scores[sim_term] = rel_score
        sim_terms = sorted(term_scores.items(), reverse=True, key=lambda item: item[1])
    q_terms = set(query.split(" "))
    vector_list = []
    for q_term in q_terms:
        if q_term not in model.wv.vocab:
            continue
        vector = model.wv.get_vector(q_term)
        vector_list.append(vector)

    extended_terms_list = []
    extended_terms = ""
    # temp_vector = vector_list
    temp_vector = [mean_vector]
    cluster_score = 99999
    try:
        for key, _ in sim_terms:
            if key not in model.wv.vocab:
                continue
            temp_vector.append(model.wv.get_vector(key))
            if len(temp_vector) < 2:
                continue
            X = scale(temp_vector)
            try:
                tend_score = hopkins(X, len(temp_vector))
            except RuntimeWarning:
                del temp_vector[len(temp_vector)-1]
                continue
            if tend_score <= cluster_score:
                cluster_score = tend_score
            else:
                del temp_vector[len(temp_vector)-1]
                continue
            extended_terms = extended_terms  + " "+ key 
    except Exception:
        print('error')
    extended_terms_list.append(extended_terms)       
    return extended_terms_list