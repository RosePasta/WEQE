import os

import rose_util.text_util as text_util


def get_bug_types(path):
    f = open(path, "r", encoding="utf8")
    bug_types = {}
    for line in f.readlines():
        tokens = line.replace("\n","").split(",")
        bug_id = tokens[1]
        bug_type = tokens[2]
        bug_types[bug_id] = bug_type
    f.close()
    return bug_types

def get_query(path):
    queries = os.listdir(path)
    summary_dict = {}    
    query_dict = {}    
    for query in queries:
        bug_id = query.replace(".txt","")
        f = open(path+query, "r", encoding="utf8")
        summary = f.readlines()[0].replace("\n","")
        summary_dict[bug_id] = text_util.remove_double_spaces(summary).strip()
        f.close()

        f = open(path+query, "r", encoding="utf8")
        text = (' '.join(f.readlines())).replace("\n"," ")
        query_dict[bug_id] = text_util.remove_double_spaces(text).strip()
        f.close()
    return summary_dict, query_dict

def get_gtf(path):
    gtfs = os.listdir(path)
    gtf_dict = {}
    for gtf in gtfs:
        bug_id = gtf.replace(".txt","")
        if bug_id.find("_mth") > -1:
            continue
        gtf_list = []
        f = open(path+gtf, "r", encoding="utf8")
        for line in f.readlines():
            line = line.replace("\n","")
            if len(line) == 0:
                continue
            gtf_list.append(line.strip())
        f.close()
        gtf_dict[bug_id] = gtf_list
    return gtf_dict

def get_files(path):
    sfiles = os.listdir(path)
    file_dict = {}
    for sfile in sfiles:
        sfile_id = sfile.replace(".txt","")
        f = open(path+sfile, "r",encoding="utf8")
        text = (' '.join(f.readlines())).replace("\n"," ")
        file_dict[sfile_id] = text_util.remove_double_spaces(text).strip()
        f.close()
    return file_dict

def get_ast(path):
    asts = os.listdir(path)
    ast_dict = {}
    for ast in asts:
        sfile_id = ast.replace(".txt","")
        f = open(path+ast, "r",encoding="utf8")
        code_tokens = []
        nl_tokens = []
        for line in f.readlines():
            line_tokens = line.replace("\n","").split("\t")            
            node_type = line_tokens[0]
            tokens = line_tokens[1].strip().split(" ")
            if node_type == "COMMENTS":
                nl_tokens += tokens
            else:
                if node_type.find("SIG") > -1:
                    code_tokens += tokens
                if node_type =="CLASSES":
                    code_tokens += tokens
        token_dict = {}
        token_dict["nl"] = ' '.join(list(set(nl_tokens)))
        token_dict["code"] = ' '.join(list(set(code_tokens)))
        ast_dict[sfile_id] = token_dict
        f.close()
    return ast_dict

def get_file_key_dict(path):
    f = open(path, "r", encoding="utf8")
    key_name_dict = {}
    name_key_dict = {}

    for line in f.readlines():
        tokens = line.replace("\n","").split(":")
        sf_id = tokens[0]
        sf_name = tokens[1]
        key_name_dict[sf_id] = sf_name
        name_key_dict[sf_name] = sf_id
    return key_name_dict, name_key_dict


def load_data(base_path, project, version, stem_type):        
    bug_path = base_path + "bugs_pp"+stem_type+"\\"+project+"\\"+version+"\\"
    summaries, bugs = get_query(bug_path)

    gtf_path = base_path + "buggy_files_index\\"+project+"\\"+version+"\\"
    gtfs = get_gtf(gtf_path)

    file_path = base_path + "files_pp"+stem_type+"\\"+project+"\\"+version+"\\"
    sfiles = get_files(file_path)

    ast_path = base_path + "ast_pp"+stem_type+"\\"+project+"\\"+version+"\\"
    file_asts = get_ast(ast_path)

    file_key_path = base_path + "fileKeyMap\\"+project+"\\"+version+".txt"
    key_name_dict, name_key_dict = get_file_key_dict(file_key_path)

    return summaries, bugs, gtfs, sfiles, file_asts, key_name_dict, name_key_dict


def classify(file_name):
    if file_name.find(".test") > -1:
        return "tf"
    if file_name.find("/test") > -1:
        return "tf"
    if file_name.find("Test") > -1:
        return "tf"
    if file_name.find("test") > -1:
        return "af"
    return "pf"
