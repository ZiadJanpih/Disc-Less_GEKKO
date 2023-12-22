import numpy as np
from Model import *
import pandas as pd 
from gekko import GEKKO
import time
import json
import os
import re
import itertools

#======================================================================================================================
#================================================== Functions =========================================================

#load the columns informations into an OOP structure 
def get_columns(dataset):
    columns=[]
    col_names=dataset.columns.values.tolist()
    for col_name in col_names:
        column = dataset[col_name]
        try:
            c= Column(name=col_name,c_type=type(column.loc[0]),min_value=column.min(),max_value=column.max())
            columns.append(c)
        except Exception as error:
            Zx=0
    return columns

# To search in the columns and retrieve the one that matches the name. 
def get_column_by_name(column_list, name):
    column=None
    for c in column_list:
        if (c.name==name):
            column=c
    return column
# To search in the columns and retrieve the one that matches the id.
def get_column_by_id(column_list, id):
    column=None
    for c in column_list:
        if (c.c_id==id):
            column=c
    return column

# create operators and load them into an OOP structure 
def get_operators():
    operators=[]
    operators.append(Operator(operator="==",o_type="E"))
    operators.append(Operator(operator=">=",o_type="LGE"))
    operators.append(Operator(operator=">",o_type="LG"))
    operators.append(Operator(operator="<=",o_type="LSE"))
    operators.append(Operator(operator="<",o_type="LS"))
    return operators

# To search in the operators and retrieve the one that matches the name.
def get_operator_by_name(operator_list, name):
    operator=None
    for o in operator_list:
        if (o.operator==name):
            operator=o
    return operator  
# To search in the operators and retrieve the one that matches the id.
def get_operator_by_id(operator_list, id):
    operator=None
    for o in operator_list:
        if (o.o_id==id):
            operator=o
    return operator 

# To search in the predicates and retrieve the one that matches the id.
def get_predicat_by_id(preds, id):
    pred=[p for p in preds if p.p_id ==id][0]
    return pred 

# calculate the min & max values for a selection condition
def getPredicateRange(p):
    cmnv = float(p.column.min_value)
    cmxv = float(p.column.max_value)
    scmnv,scmxv = None,None
    operrator= p.operator.operator
    if (operrator == ">="):
        scmnv=float(p.value)
        scmxv=cmxv
    if (operrator == "<="):
        scmxv=float(p.value)
        scmnv=cmnv
    return scmnv,scmxv    


# calculate the min & max values of the search sapce for a selection condition
def getPredicateSearchSpace(p):
    cmnv = float(p.column.min_value)
    cmxv = float(p.column.max_value)
    scmnv,scmxv = None,None
    operrator= p.operator.operator
    if (operrator == ">="):
        scmnv=cmnv
        scmxv=float(p.value)
    if (operrator == "<="):
        scmxv=cmxv
        scmnv=float(p.value)
    return scmnv,scmxv

# Retrieve the data by applying a set of predicates to a dataset.
def get_predicates_res(preds,dataset):
    pred=preds[0]
    conditions=pred.column.name+' '+pred.operator.operator+' '+str(pred.value)
    for i in range (1,len(preds),1) :
        pred=preds[i]
        conditions=conditions +' & '+pred.column.name+' '+pred.operator.operator+' '+str(pred.value)
    res=dataset.query(conditions, inplace = False)
    return res
# Check if a predicate exists in a list 
def check_predicate_exist(column,operator,value, p_list):
    for p in p_list:
        if (column == p.column and operator== p.operator and value==p.value):
            return True
    return False

# To generate predicates, specifically designed to distribute data uniformly across different intervals.
# This algorithm works by dividing the data into multiple segments of approximately equal size 
# based on the distribution of values within a specific column.
def Generate_candidate_predicates_E_depth(bin_count,Sc,Dataset):
    candidate_predicates=[]
    for c in Sc :    
        if c.operator.operator in ['<', '<=']:   
            condition=c.column.name+' >= '+str(c.value)
            c_res=Dataset.query(condition, inplace = False)
        else:
            condition=c.column.name+' <= '+str(c.value)
            c_res=Dataset.query(condition, inplace = False)
        if c_res[c.column.name].unique().size > 1:
            if bin_count >= c_res[c.column.name].unique().size :
                dim_bin=c_res[c.column.name].unique()
            else:
                _,dim_bin = pd.qcut(c_res[c.column.name], q = bin_count-1, duplicates='drop', retbins=True)
            dim_bin = dim_bin.tolist()
        else:
            dim_bin = [float(c.value), max(c.column.max_value, float(c.value))]
        if len(set(dim_bin)) == 1:
            dim_bin = dim_bin + dim_bin
        dim_bin.sort()
        for b in range(0,len(dim_bin)):
            candidate_predicates.append(Predicate(column=c.column,operator=c.operator,value=dim_bin[b],category= c.category))
                              
    return candidate_predicates 

# To generate predicates that divides a dataset into intervals of equal width 
# based on the range of values within a specific column
def Generate_candidate_predicates_E_width(bin_count,Sc,Dataset):
   
    candidate_predicates=[]
    for c in Sc :
        condition=c.column.name+' '+c.operator.operator+' '+str(c.value)
        c_res=Dataset.query(condition, inplace = False)
        search_data=pd.merge(Dataset, c_res, how="outer", indicator=True).query('_merge=="left_only"')
        distinct_val = len(search_data[c.column.name].unique())
        n_bin_attr = min(bin_count, distinct_val)
        if c.operator.operator in ['<', '<=']:
            step = (c.column.max_value -c.value) / (n_bin_attr - 1)
            candidate_predicates.append(c)
            for i in range(1, n_bin_attr):
                pred_value =c.value+ step * i
                if (check_predicate_exist(column=c.column,operator=c.operator,value=pred_value,p_list=candidate_predicates)==False):
                    candidate_predicates.append(Predicate(column=c.column,operator=c.operator,value=pred_value,category= c.category))                
        else:
            step = (c.value - c.column.min_value) / (n_bin_attr - 1)
            candidate_predicates.append(c)
            for i in range(1, n_bin_attr):
                pred_value =c.value - step * i
                if (check_predicate_exist(column=c.column,operator=c.operator,value=pred_value,p_list=candidate_predicates)==False):
                    candidate_predicates.append(Predicate(column=c.column,operator=c.operator,value=pred_value,category= c.category))                     
    return candidate_predicates    

# this function is used to calculate the probability and probability distribution for a transformation
# which may contain many predicates
def get_transformation_prob_distrib(p_ids,p_list,target_column,Dataset):
    preds=[]
    pred=[p for p in p_list if p.p_id==p_ids[0]][0]
    
    preds.append(pred)
    conditions=pred.column.name+' '+pred.operator.operator+' '+str(pred.value)
    for i in range (1,len(p_ids),1) :
        pred= [p for p in p_list if p.p_id==p_ids[i]][0]
        preds.append(pred)
        conditions=conditions +' & '+pred.column.name+' '+pred.operator.operator+' '+str(pred.value)
    res=Dataset.query(conditions, inplace = False)

    data_size=len(Dataset)
    counts=res[target_column.name].value_counts()
    new_values=res[target_column.name].value_counts().index.tolist()
    org_values=Dataset[target_column.name].value_counts().index.tolist()
    
    bpp={}
    bc={}
    for value in org_values :
        if value in new_values:
            bc[value]=int(counts[value])
            bpp[value]=round(counts[value]/Dataset[target_column.name].value_counts()[value],6)
        else:
            bc[value]=0
            bpp[value]=0
    probability = round(len(res)/data_size,6)
    cardinality= len(res)
    return preds,probability,bpp,bc,cardinality    

# find all the possible tansformations ;Examples  A>=5 ; A>=5 & B >=20 ; A >=5 & B>=20 & C <= 10 . . . .    
def Generate_transformations( p_list ,target_column, Dataset,k =2):
    groups = set(map(lambda x:x.category, p_list))
    newlist = [[y.p_id for y in p_list if y.category==x] for x in groups]
    if k < 1 :
        k=1
    if k > len(groups):
        k = len(groups)
    transformation_ids=[]
    for p in p_list:
        transformation_ids.append([p.p_id])
    for ki in range(2, k+1, 1):
        group_combinations=list(itertools.combinations(newlist, ki))        
        for gc in group_combinations :
            transformation_ids +=list(itertools.product(*gc))
    for i in range(len(transformation_ids)):
        transformation_ids[i]=list(transformation_ids[i])
        
    transformations=[]
    
    for p_ids in transformation_ids :
        predicates,probability,bpp,bc,cardinality=get_transformation_prob_distrib(p_ids,p_list,target_column,Dataset)
        transformations.append(Transformation(predicates,probability,bpp,bc,cardinality))
    return transformations
   

def initialize_variabels(column_list,operato_list,query,dataset,bin_count,fairness_type="O_FC"):
    target_column=get_column_by_name(column_list,query["Z_CC"]["column"])
    target_col_counts=dataset[target_column.name].value_counts()
    target_col_bins=dataset[target_column.name].value_counts().index.tolist()
    
    Sc=[]
    for i,p in enumerate(query["Predicates"]) :
        Sc.append(Predicate(column=get_column_by_name(column_list,p["att"]),operator=get_operator_by_name(operato_list,p["op"]),value = p["value"],category='g'+str(i)))
    Sc_res=get_predicates_res(Sc,dataset)
    Sc_counts=Sc_res[target_column.name].value_counts()
    cov_constraint=query["Z_CC"]

    Td=query[fairness_type]['bins'] # Target distribution for the target column
    od={} # original distribution for the target column in the dataset
    q_od={} # query distribution for the target column
    for i,value in enumerate(target_col_bins)  :
        od[value]=round(target_col_counts[value]/len(dataset),6)
        q_od[value]=round(Sc_counts[value]/len(Sc_res),6)
    bin0_idx=np.argmin([Td[b] - q_od[b] for b in target_col_bins])
    bin0=target_col_bins[bin0_idx]
    tuple_count=Sc_counts[bin0]//(Td[bin0])
    tc={}
    tdml=0 # calculate taget distribution maximal loss
    for value in target_col_bins :
        tc[value]=(Td[value])*tuple_count
        tdml+=max((target_col_counts[value]-tc[value]),tc[value])
    c=0.000001
    variabels={
        "target_column":target_column,
        "target_col_counts":target_col_counts,
        "target_col_bins":target_col_bins,
        "od":od,
        "q_od":q_od,
        "tc":tc,
        "td":Td,
        "tdml":tdml,
        "c":c,
        "Sc":Sc,
        "Sc_res":Sc_res,
        "Sc_counts":Sc_counts,
        "cov_constraint":cov_constraint
    }
    return variabels

def get_predicate_transformation(predicate,transformations):
    pt=None
    for t in transformations :
        if (len(t.predicates) == 1  and t.predicates[0].p_id == predicate.p_id):
            pt=t
    return pt

def get_pred_sets(transformations):
    pred_sets=[]
    for t in transformations :
        if (len(t.predicates) > 1 ):
            pred_set=[]
            for p in t.predicates:
                pred_transformation= get_predicate_transformation(p,transformations)
                if pred_transformation :
                    pred_set.append(transformations.index(pred_transformation))
            pred_set.append(transformations.index(t))
            pred_sets.append(pred_set)
    return pred_sets

def get_selectionSets(p_list,transformations):
    sel_sets=[]
    groups = set(map(lambda x:x.category, p_list))
    newlist = [[y for y in p_list if y.category==x] for x in groups]
    for g in newlist:
        sel_set=[]
        for t in transformations :
            if (len(t.predicates) > 1 ):
                for p in t.predicates:
                    if (p in g):
                        sel_set.append(transformations.index(t))
        for p in g:
            t= get_predicate_transformation(p,transformations)
            sel_set.append(transformations.index(t))
        sel_sets.append(sel_set)
    return sel_sets

def get_filtered_transformation_Threshold(transformations,pdp):
    filtered_transformation=[]
    dep_transformation=[]
    for t in transformations:
        if (len(t.predicates) == 1 ):
            filtered_transformation.append(t)
        else:
            pp_product=np.prod([get_predicate_transformation(p,transformations).probability for p in t.predicates ])
            if (abs(pp_product-t.probability) > pdp):
                dep_transformation.append(t)
    return filtered_transformation+dep_transformation

def get_filtered_transformation_TopPercent(transformations,pdp):
    filtered_transformation=[]
    transformation_to_Sort=[]
    for t in transformations:
        if (len(t.predicates) == 1 ):
            filtered_transformation.append(t)
        else:
            pp_product=np.prod([get_predicate_transformation(p,transformations).probability for p in t.predicates ])
            dep=abs(pp_product-t.probability)
            if (dep > 0):
                transformation_to_Sort.append((dep,t))
    transformation_to_Sort.sort(key=lambda y: y[0],reverse=True)
    filtered_count=int(len(transformation_to_Sort) * pdp)
    filtered_transformation=filtered_transformation+[transformation_to_Sort[t][1] for t in range(filtered_count)]

    return filtered_transformation



def query_to_prep(q):
    in_m = re.match(r'.*FROM\s(.*)\sWHERE(.*)', q)
    Predicates = []
    for cond in re.split(' AND | OR ', in_m.group(2)):
        c = re.match(r'([a-zA-Z\d\_]+)\s*(\<|\>|\<\=|\>\=)\s*([\-]?[\d]+[\.]?[\d]*)', cond.strip())
        Predicates.append(
        {
            'att': c.group(1).strip(),
            'op': c.group(2).strip(),
            'value':float( c.group(3).strip())
        })    
    return Predicates


#======================================================================================================================



def const_processing(vars,transformations,dataset,dep_sets,sel_sets,is_remote,CC_soft):
    all_dep_sets=dep_sets+sel_sets
    con_find_solution_ST=time.time()
    z = GEKKO(remote=is_remote)
    z.options.SOLVER=1  # APOPT is an MINLP solver
      # set the solver options 
    z.solver_options = ['minlp_gap_tol 0.001',\
    'minlp_maximum_iterations 10000',\
    'minlp_max_iter_with_int_sol 1000']
    #"""
    target_col_bins=vars["target_col_bins"]
    data_size=len(dataset)
    Sc_size=len(vars["Sc_res"])
    cov_constraint=vars["cov_constraint"]
    weights=vars["weights"]
    od=vars["od"]
    c=vars["c"]
    preds=[z.Var(0,lb=0,ub=1,integer=True) for i in range(len(transformations))] 
    Td=[vars["td"][b]for b in target_col_bins]
    bc=z.Array(z.Var,(len(target_col_bins)))
    CC_index=target_col_bins.index(cov_constraint["bin"])
    
    for j,b in enumerate(target_col_bins):
        z.Equation(bc[j]==data_size * od[b] * z.exp(z.sum([p * z.log( transformations[i].bpp[b]+c ) for i,p in enumerate(preds) ])))            
    z.Equations([z.sum([preds[i] for i in z_dep_set])<=1  for z_dep_set in all_dep_sets ])
    z.Equation(z.sum(preds) >=1)
    ccl=z.max3(0 ,(cov_constraint["value"]- bc[CC_index] )/cov_constraint["value"])
    if (CC_soft!=True  and weights['Ccw'] >0 ) :
        z.Equation(ccl<=0)
    Fcl=  (z.sum([z.abs2(Td[i] - (bc[i]/z.sum(bc))) for i in  range(len(Td))])/len(Td))
    minimal=(z.abs2(z.sum(bc)-Sc_size)/data_size)
    obj=(( weights['Fcw']* Fcl)+(  weights['Ccw']*ccl)+( weights['Mw']* minimal))*10
    z.Minimize(obj)
    z.solve(disp=False)
    con_find_solution_ET=time.time()
    con_find_solution_T=con_find_solution_ET - con_find_solution_ST    
    Sc_new=[]
    for i,p in enumerate(preds) :
        if (p.value[0] >0):
            trans=transformations[i]
            for t in trans.predicates :
                Sc_new.append(t)        
    res_bc=[b.value[0] for b in bc]
    res_ccl=max(0 ,(cov_constraint["value"]- res_bc[CC_index] )/cov_constraint["value"])
    res_tdl=sum([abs(Td[i] - (res_bc[i]/sum(res_bc))) for i in  range(len(Td))])/len(Td)
    res_minimal=abs(sum(res_bc)-Sc_size)/data_size
    obective={
        "Fcl":   round(res_tdl,6),
        "FclX":  round(weights['Fcw']*res_tdl,6),
        "ccl":round(res_ccl,6),
        "cclX":round(weights['Ccw']*res_ccl,6),
        "minimal":round(res_minimal,6),
        "minimalX":round(weights['Mw']*res_minimal,6),
        "Objective":round(z.options.objfcnval*1,6)
    }
    return Sc_new,con_find_solution_T,res_bc,obective


def find_constraints_solution(column_list,operato_list,queries,bin_algo,bin_counts,filter_algo,filter_values,dataset,
file_name,CC_types,is_remote=False,fairness_type="O_FC",k=2):
    print("--- Constraint Model Started ---"+str(time.asctime( time.localtime(time.time()) )))
    for query in queries["Queries"] :
        try:
            for bin_count in bin_counts:
                query["Predicates"]=query_to_prep(query["query"])
                vars=initialize_variabels(column_list,operato_list,query,dataset,bin_count,fairness_type)
                con_pre_processing_ST= time.time()
                pred_list=bin_algo(bin_count,vars["Sc"],dataset)
                all_transformations=Generate_transformations( p_list = pred_list,target_column=vars["target_column"],Dataset=dataset,k=k)
                con_pre_processing_ET= time.time()
                pre_processing_T=con_pre_processing_ET - con_pre_processing_ST
                for filter_value in filter_values :
                    con_filtering_ST= time.time()
                    transformations=filter_algo(all_transformations,filter_value)
                    con_filtering_ET= time.time()
                    con_filtering_T=con_filtering_ET - con_filtering_ST            
                    dep_sets=get_pred_sets(transformations)
                    sel_sets=get_selectionSets(pred_list,transformations)
                    Filtering_Algorithm="Top_Percentage"
                    if(filter_algo == get_filtered_transformation_Threshold):
                        Filtering_Algorithm="Top_values"   

                    for CC_type in CC_types:
                        try:
                            vars["weights"]= queries["Weights"][query["type"]] 
                            Sc_new,con_find_solution_T,bc,obj=const_processing(vars,transformations,dataset,dep_sets,sel_sets,is_remote,CC_type)
                        except Exception as error:
                            #raise error
                            if (is_remote):
                                continue
                            print('Local Gekko local Exception' + repr(error))
                            Sc_new,con_find_solution_T,bc,obj=const_processing(vars,transformations,dataset,dep_sets,sel_sets,is_remote,CC_type)
                        const_result={
                            "dataset":dataset,
                            "query":query,
                            "bin_count":bin_count,
                            "Sc_new":Sc_new,
                            "bc":bc,
                            "Sc":vars["Sc"],
                            "pred_list":pred_list,
                            "transformations":transformations,
                            "vars":vars,
                            "fairness_type":fairness_type,
                            "pre_processing_T":pre_processing_T,
                            "con_filtering_T":con_filtering_T,
                            "find_solution_T":con_find_solution_T,
                            "Filtering_Algorithm":Filtering_Algorithm,
                            "filter_value":filter_value,
                            "obj":obj
                        }
                        type="_HC"
                        if(CC_type):
                            type="_SC"
                        file=file_name+type                
                        write_result(const_result,file)
                        print("---- "+query["Istance"]+"-"+type+" - "+str(bin_count)+"B -- Is Done ---"+str(time.asctime( time.localtime(time.time()) )))
        except Exception as error:
            raise error
            print('Model Exception -- Q -' +query["Istance"]+"  --  "+ repr(error))
    print("--- Constraint Model Finished ---"+str(time.asctime( time.localtime(time.time()) )))
    return 0



def write_result(result,file_name):
    vars=result["vars"]
    Sc_size=len(vars["Sc_res"])
    result_header=['Istance',
    'Bins',
    'Condition_num',
    'Traget_Column',
    'Original_Q',
    'Origi_Q_Distrib',
    'Original_Q_card',
    'Original_SA_card',
    'CC_bin',
    'CC_value',
    "CC_percent",
    'fairness_type',
    'fairness_bins',
    "Input_prep",
    "New_Q",
    "New_Q_est_card",
    "New_Q_real_card",
    "New_Q_est_SA_card",
    "real_SA_card",
    "New_Fiarness_Bins",
    "Output_prep",
    "Objectives",
    "Filtering Algorithm",
    "Filtering value",
    "Filtering Time",
    "single_pred_count",
    "pred_sets_count",
    "PreProcessing_time",
    "Processing_time"]
    sub_result=[]
    original_Q=""
    for i,p in enumerate(result["query"]["Predicates"]) :
        if (i < len (result["query"]["Predicates"])-1):
            original_Q+=str(p["att"])+"  "+ str(p["op"])+"  "+str(p["value"]) +" AND  "
        else:
            original_Q+=str(p["att"])+"  "+ str(p["op"])+"  "+str(p["value"]) 
    
    new_Q=""
    Sc_new= result["Sc_new"]
    for p in Sc_new :                  
        new_Q+="  "+str(p.column.name)+"  " +str(p.operator.operator)+" " +str(p.value) +" AND"
    output_prep=[]
    input_prep=[]
    for sel_con in vars["Sc"]:
        pred={
            "attr": sel_con.column.name,
            "op": sel_con.operator.operator,
            "val": float(sel_con.value),
            "min": float(sel_con.column.min_value),
            "max": float(sel_con.column.max_value)
        }
        input_prep.append(pred)
    for sel_con in Sc_new:
        pred_bins=[]
        for p in result["pred_list"]:
            if (p.category == sel_con.category):
                pred_bins.append(float(p.value))
        pred={
            "attr": sel_con.column.name,
            "op": sel_con.operator.operator,
            "val": float(sel_con.value),
            "min": float(sel_con.column.min_value),
            "max":  float(sel_con.column.max_value),
            "bins": pred_bins,
            "min_bin": min(pred_bins),
            "max_bin": max(pred_bins)
        }
        output_prep.append(pred)
    Sc_new_res=get_predicates_res(Sc_new,result["dataset"])
    Sc_new_counts=Sc_new_res[vars["target_column"].name].value_counts()
    Origi_Q_Distrib={}
    for bin in vars["Sc_res"][vars["target_column"].name].value_counts().index.tolist():
        Origi_Q_Distrib[bin]=round( vars["Sc_counts"][bin]/len(vars["Sc_res"]) ,4)                  
    new_fairness={}
    for bin in Sc_new_res[vars["target_column"].name].value_counts().index.tolist():
        new_fairness[bin]=round( Sc_new_counts[bin]/len(Sc_new_res) ,4)
    CC_index=vars["target_col_bins"].index(vars["cov_constraint"]["bin"])
    sub_result.append('{:03}'.format(result["query"]["Istance"]))
    sub_result.append(result["bin_count"])
    sub_result.append(len(vars["Sc"]))
    sub_result.append(vars["target_column"].name)
    sub_result.append(original_Q)
    sub_result.append(json.dumps(Origi_Q_Distrib))
    sub_result.append(Sc_size)
    sub_result.append(vars["Sc_counts"][vars["cov_constraint"]["bin"]])
    sub_result.append(vars["cov_constraint"]["bin"])
    sub_result.append(vars["cov_constraint"]["value"])
    sub_result.append(vars["cov_constraint"]["percent"])
    sub_result.append(result["fairness_type"])
    sub_result.append(result["query"][result["fairness_type"]])
    sub_result.append(json.dumps(input_prep))            
    sub_result.append(new_Q[:-3])                               # New_Q constraint
    sub_result.append(sum(b for b in result["bc"]))             # estimated cardinality
    sub_result.append(len(Sc_new_res))                          # real cardinality    
    sub_result.append(result["bc"][CC_index])                    # estimated SA_cardinality
    sub_result.append(Sc_new_counts[vars["cov_constraint"]["bin"]])     # real SA_cardinality
    sub_result.append(json.dumps(new_fairness))  #json.dumps(new_fairness)
    sub_result.append(json.dumps(output_prep))
    sub_result.append(json.dumps(result["obj"]))
    sub_result.append(result["Filtering_Algorithm"])    
    sub_result.append(result["filter_value"]) 
    sub_result.append(result["con_filtering_T"]) 
    sub_result.append(len(result["pred_list"]))                           # single pred count
    sub_result.append(len(result["transformations"])-len(result["pred_list"]))      # pred sets count                       
    sub_result.append(result["pre_processing_T"])
    sub_result.append(result["find_solution_T"])
    con_result=[]
    con_result.append(sub_result)
    result_DF = pd.DataFrame(con_result,columns=(result_header) )
    output_path='Results\\'+file_name +'.csv'
    result_DF.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    return result_DF

if __name__ == '__main__':
    Drop_data=pd.read_csv('Data\\Dropout_data_cfu.csv') 
    column_list = get_columns(Drop_data)
    operato_list= get_operators()
    File_name="Experiment_1_"
    bins=[32]
    filter_algo=get_filtered_transformation_Threshold
    bin_algo=Generate_candidate_predicates_E_depth
    is_remote=False
    quries_file = json.load(open('Data\\DropOut_Queries.json'))
    quries=quries_file
    #quries=quries[:30]

    try:
        con_res_SH = find_constraints_solution(column_list,operato_list,quries,bin_algo,bins,filter_algo,[0.05],Drop_data,
        File_name,[True])
    except Exception as error:
        #raise error
        print('con_res_SH Exception  ' + repr(error))