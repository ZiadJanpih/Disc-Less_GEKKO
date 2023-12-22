
# This file contains presents an OOP model 
# designed for the purpose of modeling the considered optimization problem
#========================================================================================
# Operator calss : is the logical operator between conditions
class Operator(object):
    os_id=0
    def __init__(self,operator,o_type,id=None):
        if id:
            self.o_id = id
        else:
            Operator.os_id= Operator.os_id +1
            self.o_id = Operator.os_id
        self.operator = operator
        self.o_type = o_type
      
        
    def to_string(self, is_print=True):
        to_print="ID : "+str(self.o_id)+"  Operator : " +str(self.operator)+"  Type : " +str(self.o_type)
        if(is_print):
            print(to_print)
        return to_print


        
#========================================================================================
# Column calss : is the data attribute and some metadata about it
class Column(object):
    cs_id=0
    def __init__(self,name,c_type,min_value,max_value,id=None):
        if id:
            self.c_id=id
        else:
            Column.cs_id =Column.cs_id +1
            self.c_id = Column.cs_id
        self.name = name
        self.c_type = c_type
        self.min_value = min_value
        self.max_value = max_value
     
        
    def to_string(self,is_print=True):
        
        to_print="ID : "+str(self.c_id)+"  Name : " +str(self.name)+"  Type : " +str(self.c_type)+"  Min : " +str(self.min_value)+"  Max :" +str(self.max_value)
        if(is_print):
            print(to_print)
        return to_print

#========================================================================================
# Predicate calss : is a combination of Column Operator and value : attr1 >= 5 
class Predicate(object):
    ps_id=0
    def __init__(self, column,operator,value,category,id=None):
        if id:
            self.p_id=id
        else:
            Predicate.ps_id =Predicate.ps_id+1
            self.p_id = Predicate.ps_id
        self.column = column
        self.operator = operator
        self.value = value
        self.category = category

    def to_string(self,is_print=True):
        to_print="ID : "+str(self.p_id)+"  ColumnName : " +str(self.column.name)+"  Operator : " +str(self.operator.operator)+"  Value : " +str(self.value)+"  Category : " +str(self.category) 
        if(is_print):
            print(to_print)
        return to_print
    
#========================================================================================
# Transformation calss : is a set of predicates with some statistical information about the results of these predicates 
class Transformation(object):
    ts_id=0
    def __init__(self, predicates,probability,bpp, bc,cardinality,id=None):
        if id:
            self.t_id=id
        else:
            Transformation.ts_id =Transformation.ts_id+1
            self.t_id = Transformation.ts_id
        self.predicates = predicates
        self.probability = probability
        self.bpp = bpp
        self.bc = bc
        self.cardinality=cardinality
      

    def to_string(self,is_print=True):
        to_print="ID : "+str(self.t_id)+"\n"
        for p in self.predicates :
            to_print+=p.to_string(is_print=False)+"\n"
        
        to_print+="  probability : " +str(self.probability)+"\n"
        to_print+="  bpp : " +str(self.bpp)+"\n"
        to_print+="  cardinality : " +str(self.cardinality)+"\n"
        to_print+="  bc : " +str(self.bc)+"\n"
        if(is_print):
            print(to_print)
        return to_print
