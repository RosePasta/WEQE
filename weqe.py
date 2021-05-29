import os 
import sys
from rose_util import data_util, text_util, ir_util, aqe_util

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from gensim import corpora
from gensim import models
from gensim import similarities
import gensim

from sklearn.model_selection import KFold
import numpy as np
from itertools import chain, zip_longest
import random

def get_emb_model(emb_model_path):
    model = gensim.models.Word2Vec.load(emb_model_path)
    return model

def get_file_contents(project, stem_type):
    
    raw_text = []    
    version_check = set()
    versions = os.listdir(base_path+'fileKeyMap/'+project+"/")    
    target_versions = []
    for version in versions:        
        version = version.replace(".txt","")
        if len(versions)>30:
            version_tokens = version.split("_")
            version_short = version_tokens[0]+"-"+version_tokens[1]+"-"+version_tokens[2]
            if version_short in version_check:
                continue
            version_check.add(version_short)
        
        target_versions.append(version)
    for version in target_versions:
        if os.path.exists(base_path+"/ast_pp"+stem_type+"/"+project+"/"+version+"/") is False:
            continue
        sfiles = os.listdir(base_path+"/ast_pp"+stem_type+"/"+project+"/"+version+"/")
        for sfile in sfiles:
            sf_id = sfile.replace(".txt","")
            f = open(base_path+"/ast_pp"+stem_type+"/"+project+"/"+version+"/"+sfile, "r", encoding="utf8")
            code_tokens = []
            nl_tokens = []
            for line in f.readlines():
                line = line.replace("\n","")
                token_type = line.split("\t")[0]
                tokens = line.split("\t",2)[1].split(" ")                
                for token in tokens:
                    if len(token) < 2:
                        continue
                    if token_type.lower() =="comments":
                        if token not in nl_tokens:
                            nl_tokens.append(token)
                    else:
                        if token not in code_tokens:
                            code_tokens.append(token)

            if len(nl_tokens) > 0:  
                combined = [x for x in chain(*zip_longest(code_tokens, nl_tokens)) if x is not None]
                contents = ' '.join(combined)
                raw_text.append(contents)      
                for i in range(9):
                    random.shuffle(combined)
                    contents = ' '.join(combined)
                    raw_text.append(contents)

            elif len(code_tokens)> 0:
                contents = ' '.join(code_tokens)
                raw_text.append(contents)        
    raw_text = list(set(raw_text))
    train_set = [ [word for word in document.lower().split()] for document in raw_text]
    return train_set

def get_version_data(versions, stem_type):
    bug_id_set = []
    bug_version = {}
    summaries_ver = {}
    bugs_ver = {}
    gtfs_ver = {}
    file_asts_ver = {}
    file_indexes_ver = {}
    file_corpus_ver = {}

    vectorizer_ver = {}
    file_tfidf_ver = {}
    bm25_ver = {}
    dictionary_ver = {}
    lsi_ver = {}
    index_ver = {}
    cnt_vectorizer_ver = {}
    documents_term_prob_matrix_ver = {}

    bug_fixing_pairs = {}

    for version in versions:
        summaries, bugs, gtfs, sfiles, file_asts, fkey_name_dict, fname_key_dict \
            = data_util.load_data(base_path, project, version, stem_type)
        for bug_id in summaries.keys():
            bug_id_set.append(bug_id)
            bug_version[bug_id] = version
            gtf_list = gtfs[bug_id]

            # query_tokens = list(set(bugs[bug_id].split(" ")))
            query_tokens = list(set(summaries[bug_id].split(" ")))
            bug_fixing_pair = []
            buggy_file_contents = ""
            for gtf in gtf_list:
            #    buggy_file_contents += ' '.join(set(file_asts[gtf]['nl'].split(" ")))
               buggy_file_contents += ' '.join(set(file_asts[gtf]['code'].split(" ")))
            buggy_file_tokens = list(set(buggy_file_contents.split(" ")))
            combined = [x for x in chain(*zip_longest(query_tokens, buggy_file_tokens)) if x is not None]
            contents = ' '.join(combined)
            bug_fixing_pair.append(contents)
            for i in range(9):
                random.shuffle(combined)
                contents = ' '.join(combined)
                bug_fixing_pair.append(contents)
            bug_fixing_pairs[bug_id] = bug_fixing_pair
        print(version, len(bug_fixing_pairs), 'training set finish')
            
        summaries_ver[version] = summaries
        bugs_ver[version] = bugs
        gtfs_ver[version] = gtfs
        file_asts_ver[version] = file_asts

        file_indexes = []
        files_corpus = []
        for f_ind in sfiles.keys():
            files_corpus.append(sfiles[f_ind])
            file_indexes.append(f_ind)
        file_indexes_ver[version] = file_indexes
        file_corpus_ver[version] = files_corpus

        vectorizer = TfidfVectorizer(stop_words='english', max_features = 10000)
        file_tfidf = vectorizer.fit_transform(files_corpus)
        vectorizer_ver[version] = vectorizer
        file_tfidf_ver[version] = file_tfidf
        
        tokenized_corpus = [doc.split(" ") for doc in files_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_ver[version] = bm25

        texts = [ [word for word in document.lower().split()] for document in files_corpus]
        texts = [[token for token in text] for text in texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lsi = models.LsiModel(corpus, id2word=dictionary)
        index = similarities.MatrixSimilarity(lsi[corpus])
        dictionary_ver[version] = dictionary
        lsi_ver[version] = lsi
        index_ver[version] = index

        # For Jensen Shannon Model
        cnt_vectorizer, documents_term_prob_matrix = ir_util.get_doc_term_probs_jsm(files_corpus)
        cnt_vectorizer_ver[version] = cnt_vectorizer
        documents_term_prob_matrix_ver[version] = documents_term_prob_matrix
        print(version, 'load finish')
    return bug_id_set, bug_version, bug_fixing_pairs, summaries_ver, bugs_ver, gtfs_ver, file_asts_ver, file_indexes_ver, file_corpus_ver, \
        vectorizer_ver, file_tfidf_ver,bm25_ver, dictionary_ver, lsi_ver, index_ver, cnt_vectorizer_ver, documents_term_prob_matrix_ver


so_emb_model_path = "./dataset/embedding_models/so_model/so_java.model"
scor_emb_model_path = "./dataset/embedding_models/scor_model/model.output"

base_path = "./dataset/"
base_bug_path = base_path+"bugs_pp/"

ir_models = ['vsm','bm','lsa','jsm']
extended_term_num_list = [10, 20, 30, 40, 50, 75, 100]
num_top_index = 10

# weqe_types = ['so','scor']
weqe_types = ['so','scor']
tuning_type = 'bug' #'model','all','bug','file'

for weqe_type in weqe_types:
    projects = os.listdir(base_bug_path)
    stem_type = ''
    emb_model_path = so_emb_model_path
    if weqe_type =="scor":
        stem_type ='_stem'
        emb_model_path = scor_emb_model_path

    projects = ['roo']
    total_bugs = 0
    total_mrr = 0
    for project in projects:
        print(weqe_type, stem_type, project,emb_model_path)
        bug_type_path = base_path +"bug_types/"+project+".csv"
        bug_types = data_util.get_bug_types(bug_type_path)
        
        rr_list = [0] * 4
        bug_num = 0
        versions = os.listdir(base_bug_path+project+"/")

        file_train_set = []
        file_train_set = get_file_contents(project, stem_type)
        print(len(file_train_set), len(file_train_set[0]), 'file trainin set')

        bug_id_set, bug_version, bug_fixing_pairs, summaries_ver, bugs_ver, gtfs_ver, file_asts_ver, file_indexes_ver, file_corpus_ver, \
            vectorizer_ver, file_tfidf_ver, \
            bm25_ver, \
            dictionary_ver, lsi_ver, index_ver, \
            cnt_vectorizer_ver, documents_term_prob_matrix_ver = get_version_data(versions, stem_type)

        cv = KFold(n_splits=10,shuffle=False)
        train_iter = 0            
        for train_iter, (train_index, test_index) in enumerate(cv.split(bug_id_set)):            
            train_bugs = (np.array(bug_id_set)[train_index.astype(int)]).tolist()
            test_bugs = (np.array(bug_id_set)[test_index.astype(int)]).tolist()

            train_contents = []
            for bug_id in train_bugs:
                if bug_id not in bug_fixing_pairs.keys():
                    continue
                for bug_fixing_pair in bug_fixing_pairs[bug_id]:
                    train_contents.append(bug_fixing_pair)
            # train_set = [ list(set([word for word in document.lower().split()])) for document in train_contents]            
            train_set = [[word for word in document.lower().split()] for document in train_contents]            
            print(train_iter, len(train_set), "train start")            
            if tuning_type =="all":
                train_set = train_set + file_train_set
            elif tuning_type =="file":
                train_set = file_train_set

            print(train_iter, len(train_set), "train start")
            model = get_emb_model(emb_model_path)
            if tuning_type != "model":
                model.build_vocab(train_set, update=True)
                model.train(train_set, total_examples=model.corpus_count, epochs=10, start_alpha= 0.0025)
                print(train_iter, "train finish")

            
            for bug_id in test_bugs:
                buggy_version = bug_version[bug_id]
                summary = summaries_ver[buggy_version][bug_id]
                bug_report = bugs_ver[buggy_version][bug_id]
                gtf_list = gtfs_ver[buggy_version][bug_id]
                file_indexes = file_indexes_ver[buggy_version]
                files_corpus= file_corpus_ver[buggy_version]
                bug_type = bug_types[bug_id]

                cnt_vectorizer = cnt_vectorizer_ver[buggy_version]
                extends_query = aqe_util.get_we_query(summary, model, "cent",extended_term_num_list, extended_term_num_list[-1])

                vectorizer = vectorizer_ver[buggy_version]
                file_tfidf = file_tfidf_ver[buggy_version]
                bm25 = bm25_ver[buggy_version]

                dictionary = dictionary_ver[buggy_version]
                lsi = lsi_ver[buggy_version]
                index = index_ver[buggy_version]
                documents_term_prob_matrix = documents_term_prob_matrix_ver[buggy_version]                
                for ir in ir_models:
                
                    print(stem_type, ir, project, bug_id, end="\t")

                    write_f_rr = open("./results/rr_"+project+"_"+tuning_type+"_"+weqe_type+".txt", "a", encoding="utf8")
                    write_f_rr.write(ir+"\t"+stem_type+"\t"+project+"\t"+bug_id+"\t"+bug_type+"\t")
                    write_f_ap = open("./results/ap_"+project+"_"+tuning_type+"_"+weqe_type+".txt", "a", encoding="utf8")
                    write_f_ap.write(ir+"\t"+stem_type+"\t"+project+"\t"+bug_id+"\t"+bug_type+"\t")

                    len_ind = 0
                    for scor_query in extends_query:
                        scor_query = bug_report+" "+scor_query
                        scor_query = text_util.remove_double_spaces(scor_query).strip()
                        
                        ext_query_num = extended_term_num_list[len_ind]
                        write_f_query = open("./results/query/"+project+"_"+tuning_type+"_"+weqe_type+".txt", "a", encoding="utf8")
                        write_f_query.write(bug_id+"\t"+str(ext_query_num)+"\t"+scor_query+"\n")
                        len_ind += 1

                        if ir =="vsm":
                            sim_scores = ir_util.retrieval_tfidf(vectorizer, file_tfidf, file_indexes, scor_query)
                        elif ir =="bm":
                            sim_scores = ir_util.retrieval_bm25(bm25, file_indexes, scor_query)
                        elif ir =="lsa":
                            sim_scores = ir_util.retrieval_lsi(dictionary, lsi, index, file_indexes, scor_query) 
                        elif ir =="jsm":
                            query_term_prob_list = ir_util.get_query_term_probs_jsm(cnt_vectorizer, scor_query)
                            sim_scores = ir_util.retrieval_jsm(documents_term_prob_matrix, file_indexes, query_term_prob_list)                
                        top_rank, rr, ap, non_bfile_indexes, top_file_indexes \
                            = ir_util.evaluation(sim_scores, gtf_list, len(files_corpus), num_top_index)
                        print(top_rank, end=" ")
                        write_f_rr.write(str(rr)+"\t")
                        write_f_ap.write(str(ap)+"\t")
                    write_f_rr.write("\n")
                    write_f_rr.close()
                    write_f_ap.write("\n")
                    write_f_ap.close()
                    print()

                bug_num += 1

print(weqe_type, stem_type,tuning_type, emb_model_path)


