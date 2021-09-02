import numpy as np
import pandas as pd
from math import log, e
from tqdm import tqdm
import time, sys
from IPython.display import clear_output

class MLearningUtils():    
    
    def __init__(self):        
        self.tg_name = None
        self.att_list = None
        self.data = None        

    def check_missing_data(self, dataset: pd.DataFrame):  
        print('Qtd rows X columns:{}'.format(dataset.shape))
        print('\nAre there any Missing values?:{}'.format(dataset.isnull().values.any()))
        print('\nTotal of missing ROWS values: {}'.format(dataset.shape[0] - dataset.dropna().shape[0]))
        print('\nHas duplicated values? how to drop => df=df.drop_duplicates() \n{}'.format(dataset.duplicated().value_counts()))
        print('\nMissing values by columns:\n{}'.format(dataset.isnull().sum()))
        print('\nTotal sum of missing values: {}'.format(dataset.isnull().sum().sum()))    
        print('\nPercentage of missing values for each variable:\n{}'.format(dataset.isnull().mean()))
        print('\n')
        print(dataset.info())        
    
    def encode_cat_to_discrete_feature(self, dataframe: pd.DataFrame, cat_column_name: str, new_discrete_col_name: str):
        
        if new_discrete_col_name in dataframe.columns:
            print('column: ' + new_discrete_col_name + ' already exists')
        else:
            dataframe[cat_column_name] = dataframe[cat_column_name].astype('category')
            dataframe[new_discrete_col_name] = dataframe[cat_column_name].cat.codes            
            print('column: ' + cat_column_name + ' created')           
            
            
    def get_entropy(self, attribute: pd.Series) -> np.float64:
        
        ent = 0
        sumValues = attribute.value_counts().sum()
        #print('\nsum of {} Values: {}'.format(attribute.name, sumValues))    
        for value in attribute.value_counts():
            #print('value: {}'.format(value))
            ent -= ((value/sumValues) * log((value/sumValues), 2))
        #print('ent -= ((value/sumValues) * log((value/sumValues), 2))')
        #print('\nEntropy of {}: {}'.format(attribute.name, ent))   
        return ent             
    
    
    def get_attribute_entropy(self, dataframe: pd.DataFrame, attribute_name: str, target_name: str):
        
        attributes_values = dataframe[attribute_name].value_counts()
        #print('\nValue counts of {} by {}:'.format(attribute_name, target_name))
        #print(attributes_values)
        #print("")
        #print('Total count of {} by {}: {}'.format(attribute_name, target_name, attributes_values.sum()))    
        #print("")

        attrib_by_target_counts = dataframe.groupby([attribute_name,target_name])[attribute_name].count()
        #print('Count of {} by {}: '.format(attribute_name, target_name))
        #print(attrib_by_target_counts)
        #print("")
        
        entropy = 0
        
        for attrib_values in attributes_values.index:
            
            result = 0
            att_ratio = 0    
            count_attrib_by_target = 0
            name = ''
            temp = 0

            for att_count, value in zip(attrib_by_target_counts, attrib_by_target_counts.index):

                if attrib_values == value[0]:        
                    # entropy_mid = -( (2/4 * log(2/4, 2)) +  (2/4 * log(2/4, 2)) ) * 4/10            
                    #print('att_count: {} total: {} value: {}'.format(att_count, attributes_values.loc[value[0]], value[0] ))        
                    temp = attributes_values.loc[value[0]]
                    att_ratio = att_count / temp
                    result += (att_ratio * log(att_ratio, 2))            
                    count_attrib_by_target = temp                  
                    name = value[0]

            #print('att name:{}, entropy: {}'.format(name, ( (-result) * (count_attrib_by_target/attributes_values.sum())) ))
            entropy += ( (-result) * (count_attrib_by_target/attributes_values.sum()))

        #print('Entropy({},{}): {}'.format(target_name, attribute_name, entropy) )
        return entropy    
   
    
    def get_information_gain(self, dataframe, target_name, attribute_list = []):        
        
        self.data = dataframe
        self.tg_name = target_name    
        self.att_list = attribute_list
        
        target_attribute = []
        entropy_target = []
        entropy_attr = []
        ig_score = []

        entropy = 0        

        # Get entropy from target
        target_entropy = self.get_entropy(self.data[self.tg_name])
        #print('Target entropy: {}'.format(target_entropy))       

        if self.att_list == []:
            self.att_list =  self.data.select_dtypes('object').columns
            
        #for attribute in dataframe.select_dtypes('object').columns:
        for attribute in tqdm(self.att_list, desc="Getting entropy from dtypes('object') features"):
            
            #self.show_progress_bar(counter / len(self.att_list))
            
            name = '({},{})'.format(self.tg_name, attribute)        
            entropy = self.get_attribute_entropy(self.data, attribute, self.tg_name)        

            target_attribute.append(name)
            entropy_target.append(target_entropy)
            entropy_attr.append(entropy)
            ig_score.append(target_entropy - entropy)

            entropy = 0
            name = ''

        record = { 
            'target_attribute' : target_attribute, 
            'entropy_target': entropy_target, 
            'entropy_attr' : entropy_attr, 
            'ig_score' : ig_score 
        }    

        dframe = pd.DataFrame(record, columns = ['target_attribute', 'entropy_target', 'entropy_attr', 'ig_score'])
        return dframe.sort_values(['ig_score'],ascending=False)    