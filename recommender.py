import numpy as np
import pandas as pd
import time, sys
from IPython.display import clear_output
from tqdm import tqdm

class recommender_by_similarity():
    
    def __init__(self):        
        self.data = None
        self.user_id_column_name = None
        self.item_id_column_name = None
        self.similarity_matrix = None    
        self.columns_to_append_on_results = None
        self.qtd_max_data_items_to_compare = None
    
    def get_user_items(self, user):
        user_data = self.data[self.data[self.user_id_column_name] == user]
        user_items = list(user_data[self.item_id_column_name].unique())        
        return user_items    
    
    #Get unique users for a given item
    def get_users_by_user_item(self, item):
        item_data = self.data[self.data[self.item_id_column_name] == item]
        users_by_item = set(item_data[self.user_id_column_name].unique())
        return users_by_item    
    
    #Get unique items from dataset
    def get_all_items(self):        
        
        #Get a list of most rated items by users 
        data_grouped = self.data.groupby([self.item_id_column_name]).agg({self.user_id_column_name: 'count'}).reset_index()
        data_grouped.rename(columns = {self.user_id_column_name: 'Qtd_Users'},inplace=True)

        #Sort the items based upon recommendation score
        data_sort = data_grouped.sort_values(['Qtd_Users', self.item_id_column_name], ascending = [0,1])

        # Get top (max_unique_data_items_to_compare) items most rated by users 
        all_items = data_sort[self.item_id_column_name].head(self.max_unique_data_items_to_compare).unique()
            
        return all_items
    
    #Construct cooccurence matrix
    def generate_cooccurence_matrix(self, user_items, all_items):        
        
        # 3.1 Get users for all items in user_items.
        users_by_user_item = []        
        for i in range(0, len(user_items)):
            users_by_user_item.append(self.get_users_by_user_item(user_items[i]))
        
        # 3.2 Initialize the item cooccurence matrix of size: len(user_items) X len(all_items)
        cooccurence_matrix = np.array(np.zeros(shape=(len(user_items), len(all_items))), float)           

        # 3.3 Get similarity between user items and all unique items in the data: len(user_items) X len(all_items)
        for i in tqdm(range(0,len(all_items)), desc="Processing: "):
            
            #Get unique users of each item
            items_i_data = self.data[self.data[self.item_id_column_name] == all_items[i]]
            users_i = set(items_i_data[self.user_id_column_name].unique())            

            # Iterate for each user
            for j in range(0,len(user_items)):       

                #Get unique users of an item j(user)
                users_j = users_by_user_item[j]

                #Get intersection (which belongs to all of them) of users of items i(user_items) and j(users_by_user_item)
                users_intersection = users_i.intersection(users_j)

                #Get cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    
                    #Get union of users of items i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0

        print(cooccurence_matrix.shape)
        return cooccurence_matrix

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user_value, cooccurence_matrix, all_items, user_items):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
 
        #Sort the indices of user_sim_scores based upon their value and keep the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(user_sim_scores)), reverse=True)
        
        #for i in sort_index[0:10]:
        #    print(i)    
    
        #Create a dataframe from the following
        columns = ['user_id', self.item_id_column_name, 'score', 'rank']
        
        #index = np.arange(1) # array of numbers for the number of samples
        df_result = pd.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        counter = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_items[sort_index[i][1]] not in user_items and counter <= 10:
                df_result.loc[len(df_result)] = [user_value, all_items[sort_index[i][1]], sort_index[i][0], counter]
                counter = counter+1                
        
        #Handle the case where there are no recommendations
        if df_result.shape[0] == 0:
            print("The current user has no items for training the item similarity recommendation model.")
            return -1
        else:            
            df_Result2 = pd.merge(df_result, self.data[self.columns_to_append_on_results].drop_duplicates([self.item_id_column_name]), on=self.item_id_column_name, how="left")            
            return df_Result2    
        
    #Use item similarity model to make recommendations
    def recommend(self, user_value, user_column_name, item_column_name, dataset, columns_to_append_on_results, max_unique_data_items_to_compare):
        
        self.data = dataset
        self.user_id_column_name = user_column_name
        self.item_id_column_name = item_column_name
        self.columns_to_append_on_results = columns_to_append_on_results
        self.max_unique_data_items_to_compare = max_unique_data_items_to_compare
        
        # 1. Get all unique items from user
        user_items = self.get_user_items(user_value)         
        
        # 2. Get a list of best ranked items in order to get best similarity
        all_items = self.get_all_items()
        
        print("No. of unique items for the user: %d" % len(user_items))
        print("No. of unique items to be used for the recommendation dataset: %d" % len(all_items)) 
         
        # 3. Generate item cooccurence matrix of size: len(user_items) X len(items)        
        cooccurence_matrix = self.generate_cooccurence_matrix(user_items, all_items)     
        
        # 4. Use the cooccurence matrix to make recommendations        
        df_recommendations = self.generate_top_recommendations(user_value, cooccurence_matrix, all_items, user_items)
                
        return df_recommendations