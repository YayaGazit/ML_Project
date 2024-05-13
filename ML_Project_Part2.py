# imports
import pandas as pd
import numpy as np
import copy
import datetime
import re
#from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier



RawSet = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\FinalTrainSet2.pkl')
X = RawSet.drop('sentiment', axis=1) 
y = RawSet['sentiment'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #holdout 80 / 20 split

# ========================PART 1 Function=====================================================
# RawSet = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\XY_train.pkl')
# X = RawSet.drop('sentiment', axis=1) 
# y = RawSet['sentiment'] 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #holdout 80 / 20 split
# 
# 
# def preprocess_dataset(df):
#     
#     
#     processed_df = df.copy()
#     
# 
#     for col in processed_df.columns: #conversion every col for string
#        processed_df[col] = processed_df[col].apply(str)
#     
#     columns_to_replace_nulls = ["email", "gender", "email_verified", "blue_tick", "embedded_content", "platform"]
#     for column in columns_to_replace_nulls:
#         processed_df[column] = processed_df[column].fillna("Unknown")
#         
#     processed_df['account_creation_date'] = pd.to_datetime(processed_df['account_creation_date'])
#     processed_df['account_creation_date'] = processed_df['account_creation_date'].dt.date
#     today = datetime.date.today()  
#     delta = processed_df['account_creation_date'].apply(lambda date: (today - date).days / 365)
#     processed_df['seniority'] = delta
#     processed_df["previous_messages_dates"].apply(type).unique()
#     
#     
#  #Count number of differents dates
#     def count_expressions(text):
#         if text == []:
#             return 0
#         return len(re.findall("'[^']*'", text))
# 
#     processed_df["number_of_previous_messages"] = processed_df["previous_messages_dates"].apply(count_expressions)
#     processed_df["number_of_new_followers"] = processed_df["date_of_new_follower"].apply(count_expressions)
#     processed_df["number_of_new_follows"] = processed_df["date_of_new_follow"].apply(count_expressions)
#     
#     #Get the lastest date(year) from all the cell
#     def get_earliest_date(text):
#         if pd.isnull(text) or text == []:
#             return "Unknown"
#         dates_strings = re.findall("'(.+?)'", text)
#         if len(dates_strings) > 0:
#             dates = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dates_strings]
#             earliest_date = min(dates)
#             return earliest_date.year
#         else:
#             return "Unknown"
#     
#     processed_df["Earlier_date_of_follower"] = processed_df["date_of_new_follower"].apply(get_earliest_date)
#     processed_df["Earlier_date_of_follow"] = processed_df["date_of_new_follow"].apply(get_earliest_date)
#     
#     #Get the lastest date(year) from all the cell
#     def get_latest_date(text):
#       if pd.isnull(text) or text == []:
#           return "Unknown"
#       dates_strings = re.findall("'(.+?)'", text)
#       if len(dates_strings) > 0:
#           dates = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dates_strings]
#           latest_date = max(dates)
#           return latest_date.year
#       else:
#           return "Unknown"
#      
#     processed_df["Latest_date_of_follower"] = processed_df["date_of_new_follower"].apply(get_latest_date)
#     processed_df["Latest_date_of_follow"] = processed_df["date_of_new_follow"].apply(get_latest_date)
#     
#     #Get the email's company name
#     def get_company(email):
#         if email == "Unknown" or pd.isnull(email):
#             return "Unknown"
#         parts = re.split("@", email)
#         if len(parts) > 1:
#             company = parts[1].split(".")[0]
#             return company
#         return "Unknown"
#     processed_df["email_company"] = processed_df["email"].apply(get_company)
#     
#     ###message_year####
#     def extract_year_from_date(date):
#         try:
#             return datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").year
#         except ValueError:
#             return "Unknown"
#     
#     processed_df['message_year'] = processed_df['message_date'].apply(extract_year_from_date)
#     
#     #some text  features
#     def get_text_features(text):
#         if pd.isnull(text) or text == '':
#             return {"length": 0, "words": 0, "sentences": 0, "avg_word_length": 0}
#         num_words = len(text.split())
#         num_sentences = text.count('.') + 1
#         num_chars = len(text)
#         avg_word_length = sum(len(word) for word in text.split()) / num_words if num_words else 0
#         return {"length": num_chars, "words": num_words, "sentences": num_sentences, "avg_word_length": avg_word_length}
# 
#     text_features = processed_df["text"].apply(get_text_features)
#     processed_df["char_count"] = text_features.apply(lambda x: x["length"])
#     processed_df["word_count"] = text_features.apply(lambda x: x["words"])
#     processed_df["sentence_count"] = text_features.apply(lambda x: x["sentences"])
#     processed_df["avg_word_length"] = text_features.apply(lambda x: x["avg_word_length"]) 
#     
#     #discrization for message_time from message_date
#     def get_time_of_day(dt):
#         if pd.isnull(dt):
#             return "unknown"
#         hour = dt.hour
#         if hour >= 0 and hour < 8:
#             return "morning"
#         elif hour >= 8 and hour < 20:  
#             return "afternoon"
#         else:
#             return "evening"
# 
#     processed_df["message_date"] = pd.to_datetime(processed_df["message_date"], errors='coerce')
#     processed_df["message_time"] = processed_df["message_date"].apply(get_time_of_day)
#     
#     
#  #   texts = processed_df['text']
#  #   def vectorize_and_filter(texts, num_features):
#      #vectorizer
#    #     vectorizer = TfidfVectorizer(max_df=0.1)
#    #     vectors = vectorizer.fit_transform(texts)
#     #filter important words
#    #     indices = filter_top_words(vectors, num_features)
#    #     filtered_vectors = vectors[:, indices]
#     #series of data
#    #     filtered_df = vectors_to_df(filtered_vectors, vectorizer, indices)
#     #    return filtered_df
# 
# 
#    # def filter_top_words(vectors, N):
#    #     scores = np.asarray(vectors.sum(axis=0)).flatten()
#    #     topN = np.argsort(scores)[::-1][:N]
#    #     return topN
# 
# 
#    # def vectors_to_df(vectors, vectorizer, indices):
#    #     feature_names = vectorizer.get_feature_names_out()
#    #     words = [feature_names[i] for i in indices]
#     #    return pd.DataFrame(vectors.toarray(), columns=words)
# 
# 
#    # df = vectorize_and_filter(processed_df['text'], 30)
# 
#     ####The combination of features from TF-IDF####
#    # processed_df.reset_index(drop=True, inplace=True)
#   #  processed_df = pd.merge(processed_df, df, left_index=True, right_index=True)
#     
#     def fill_missing_gender(p_female=0.4, p_male=0.47, p_unknown=0.13):
# 
#      for idx, row in processed_df.iterrows():
#         if pd.isnull(row['gender']) or row['gender'] == 'Unknown':
#             gender = None
#             while gender is None:
#                 rnd = np.random.random()
#                 if rnd <= p_female: 
#                     gender = 'F'
#                 elif rnd > p_female and rnd < (p_female + p_male):
#                     gender = 'M'  
#                 else:
#                     gender = 'Unknown'
#                 processed_df.at[idx, 'gender'] = gender
#                 
#     fill_missing_gender()   
#                 
#                 
#     def fill_missing_email_verified(p_true=0.46, p_false=0.45, p_unknown=0.09):
# 
#      for idx, row in processed_df.iterrows():
#         if pd.isnull(row['email_verified']) or row['email_verified'] == 'Unknown':
#             email_verified = None
#             while email_verified is None:
#                 rnd = np.random.random()
#                 if rnd <= p_true:
#                     email_verified = 'True'
#                 elif rnd > p_true and rnd < (p_true + p_false):
#                     email_verified = 'False'
#                 else:
#                     email_verified = 'Unknown'
#                 processed_df.at[idx, 'email_verified'] = email_verified
#                 
#     fill_missing_email_verified()   
#                 
#     def fill_missing_blue_tick(p_true=0.63, p_false=0.25, p_unknown=0.12):
#      for idx, row in processed_df.iterrows():
#         if pd.isnull(row['blue_tick']) or row['blue_tick'] == 'Unknown':
#             blue_tick = None
#             while blue_tick is None:
#                 rnd = np.random.random()
#                 if rnd <= p_true:
#                     blue_tick = 'True'
#                 elif rnd > p_true and rnd < (p_true + p_false):
#                     blue_tick = 'False'
#                 else:
#                     blue_tick = 'Unknown'
#                 processed_df.at[idx, 'blue_tick'] = blue_tick
#                 
#     fill_missing_blue_tick()   
#                 
#     def fill_missing_embedded_content(p_false=0.55, p_mp4=0.16, p_link=0.12, p_jpeg=0.09, p_unknown=0.08):
#      for idx, row in processed_df.iterrows():
#         if pd.isnull(row['embedded_content']) or row['embedded_content'] == 'Unknown':
#             embedded_content = None
#             while embedded_content is None:
#                 rnd = np.random.random()
#                 if rnd <= p_false:
#                     embedded_content = 'False'
#                 elif rnd > p_false and rnd < (p_false + p_mp4):
#                     embedded_content = 'mp4'
#                 elif rnd > (p_false + p_mp4) and rnd < (p_false + p_mp4 + p_link):
#                     embedded_content = 'link'
#                 elif rnd > (p_false + p_mp4 + p_link) and rnd < (p_false + p_mp4 + p_link + p_jpeg):
#                     embedded_content = 'jpeg'
#                 else:
#                     embedded_content = 'Unknown'
#                 processed_df.at[idx, 'embedded_content'] = embedded_content
#                 
#     fill_missing_embedded_content()   
#                 
#     def fill_missing_platform(p_facebook=0.23, p_tiktok=0.18, p_x=0.16, p_telegram=0.15, p_instagram=0.14, p_whatsapp=0.08, p_unknown=0.06):
#      for idx, row in processed_df.iterrows():
#         if pd.isnull(row['platform']) or row['platform'] == 'Unknown':
#             platform = None
#             while platform is None:
#                 rnd = np.random.random()
#                 if rnd <= p_facebook:
#                     platform = 'facebook'
#                 elif rnd > p_facebook and rnd < (p_facebook + p_tiktok):
#                     platform = 'tiktok'
#                 elif rnd > (p_facebook + p_tiktok) and rnd < (p_facebook + p_tiktok + p_x):
#                     platform = 'x'
#                 elif rnd > (p_facebook + p_tiktok + p_x) and rnd < (p_facebook + p_tiktok + p_x + p_telegram):
#                     platform = 'telegram'
#                 elif rnd > (p_facebook + p_tiktok + p_x + p_telegram) and rnd < (p_facebook + p_tiktok + p_x + p_telegram + p_instagram):
#                     platform = 'instagram'
#                 elif rnd > (p_facebook + p_tiktok + p_x + p_telegram + p_instagram) and rnd < (p_facebook + p_tiktok + p_x + p_telegram + p_instagram + p_whatsapp):
#                     platform = 'whatsapp'
#                 else:
#                     platform = 'Unknown'
#                 processed_df.at[idx, 'platform'] = platform
#                 
#     fill_missing_platform()  
#                 
#     #sampeling emails for missing values
# #only for email_company rows with unknown
#     unknown_rows = processed_df[processed_df['email_company'] == 'Unknown']  
# 
# #Random sampling from existing values
#     sample_emails = processed_df[processed_df['email_company'] != 'Unknown'].sample(len(unknown_rows))  
# 
# #fill random sampels
#     for email, idx in zip(sample_emails, unknown_rows.index):
#        processed_df.loc[idx, 'email_company'] = email
#       
#       
#       #convert to Binary
#     mapping2gender = {'M': 1, 'F': 0}
#     processed_df['gender'] = processed_df['gender'].map(mapping2gender)
# 
#     mapping2email_verified = {'True': 1, 'False' : 0}
#     processed_df['email_verified'] = processed_df['email_verified'].map(mapping2email_verified)
# 
#     mapping2blue_tick = {'True' : 1, 'False' : 0}
#     processed_df['blue_tick'] = processed_df['blue_tick'].map(mapping2blue_tick)
#     
#     
#     # convert to NUMERIC VALUES
#     processed_df['Earlier_date_of_follower'] = pd.to_numeric(processed_df['Earlier_date_of_follower'], errors='coerce')
#     processed_df['Earlier_date_of_follow'] = pd.to_numeric(processed_df['Earlier_date_of_follow'], errors='coerce')
#     processed_df['Latest_date_of_follower'] = pd.to_numeric(processed_df['Latest_date_of_follower'], errors='coerce')
#     processed_df['Latest_date_of_follow'] = pd.to_numeric(processed_df['Latest_date_of_follow'], errors='coerce')
# 
#     
#     
#     #calculate MEAN FOR any of variable
#     mean_Earlier_date_of_follower = round(processed_df['Earlier_date_of_follower'].mean())
#     mean_Earlier_date_of_follow = round(processed_df['Earlier_date_of_follow'].mean())
#     mean_Latest_date_of_follower = round(processed_df['Latest_date_of_follower'].mean())
#     mean_Latest_date_of_follow = round(processed_df['Latest_date_of_follow'].mean())
#     
#     #Convert Unknown to null
#     processed_df['Earlier_date_of_follower'] = processed_df['Earlier_date_of_follower'].replace("Unknown", np.nan)
#     processed_df['Earlier_date_of_follow'] = processed_df['Earlier_date_of_follow'].replace("Unknown", np.nan)
#     processed_df['Latest_date_of_follower'] = processed_df['Latest_date_of_follower'].replace("Unknown", np.nan)
#     processed_df['Latest_date_of_follow'] = processed_df['Latest_date_of_follow'].replace("Unknown", np.nan)
#     
#     #Fillin missing values with the mean in each column
#     processed_df['Earlier_date_of_follower'] = processed_df['Earlier_date_of_follower'].fillna(mean_Earlier_date_of_follower)
#     processed_df['Earlier_date_of_follow'] = processed_df['Earlier_date_of_follow'].fillna(mean_Earlier_date_of_follow)
#     processed_df['Latest_date_of_follower'] = processed_df['Latest_date_of_follower'].fillna(mean_Latest_date_of_follower)
#     processed_df['Latest_date_of_follow'] = processed_df['Latest_date_of_follow'].fillna(mean_Latest_date_of_follow)
#     
#     # One_hot_encoding for embedded_content
#     embedded_content_dummies = pd.get_dummies(processed_df['embedded_content'], prefix='embedded_content')
#     platform_dummies = pd.get_dummies(processed_df['platform'], prefix='platform')
#     message_time_dummies = pd.get_dummies(processed_df['message_time'], prefix='message_time')
#     # Add to proccess_df
#     processed_df = pd.concat([processed_df, embedded_content_dummies], axis=1)
#     processed_df = pd.concat([processed_df, platform_dummies], axis=1)
#     processed_df = pd.concat([processed_df, message_time_dummies], axis=1)
#     
#     # Convert to binary
#     processed_df['embedded_content_mp4'] = processed_df['embedded_content_mp4'].replace({True: 1, False: 0})
#     processed_df['embedded_content_link'] = processed_df['embedded_content_link'].replace({True: 1, False: 0})
#     processed_df['embedded_content_False'] = processed_df['embedded_content_False'].replace({True: 1, False: 0})
#     processed_df['embedded_content_jpeg'] = processed_df['embedded_content_jpeg'].replace({True: 1, False: 0})
# 
#     processed_df['platform_whatsapp'] = processed_df['platform_whatsapp'].replace({True: 1, False: 0})
#     processed_df['platform_telegram'] = processed_df['platform_telegram'].replace({True: 1, False: 0})
#     processed_df['platform_x'] = processed_df['platform_x'].replace({True: 1, False: 0})
#     processed_df['platform_instagram'] = processed_df['platform_instagram'].replace({True: 1, False: 0})
#     processed_df['platform_facebook'] = processed_df['platform_facebook'].replace({True: 1, False: 0})
#     processed_df['platform_tiktok'] = processed_df['platform_tiktok'].replace({True: 1, False: 0})
# 
#     processed_df['message_time_morning'] = processed_df['message_time_morning'].replace({True: 1, False: 0})
#     processed_df['message_time_afternoon'] = processed_df['message_time_afternoon'].replace({True: 1, False: 0})
#     processed_df['message_time_evening'] = processed_df['message_time_evening'].replace({True: 1, False: 0})
#     
#     # Frequency Encoding for Email- too many of categories
#     email_freq = processed_df['email_company'].value_counts(normalize=True)
# #replace the values for numeric
#     processed_df['email_company'] = processed_df['email_company'].map(email_freq)
#     
#     
# 
# #Normalization of variables by min-max scaling
#     scaler = MinMaxScaler()
# 
# #pick col for scaler
#     columns_to_scale = ['seniority', 'number_of_previous_messages', 'number_of_new_followers', 
#                     'number_of_new_follows', 'char_count', 'word_count', 'sentence_count',
#                     'avg_word_length','Earlier_date_of_follower','Earlier_date_of_follow',
#                     'Latest_date_of_follower','Latest_date_of_follow','message_year']
# #Normalization
#     processed_df[columns_to_scale] = scaler.fit_transform(processed_df[columns_to_scale])
#     
#     processed_df=copy.deepcopy(processed_df)
#     processed_df = processed_df.drop(columns=['textID', 'text', 'message_date', 'account_creation_date',
#                                                       'previous_messages_dates', 'date_of_new_follower',
#                                                       'date_of_new_follow', 'email', 'embedded_content', 'platform', 'message_time'])
#     
#     return processed_df
# =============================================================================

                
# =============================================================================
# check_X_train=pd.DataFrame(preprocess_dataset(X_train))
# 
# 
# 
# 
# check_X_train=pd.DataFrame(preprocess_dataset(X_train))
# check_X_test=pd.DataFrame(preprocess_dataset(X_test)) 
# 
# 
# 
# def initialize_vectorizer(max_df=0.1):
#     return TfidfVectorizer(max_df=max_df)
# 
# def filter_top_words(vectors, num_features):
#     scores = np.asarray(vectors.sum(axis=0)).flatten()
#     topN_indices = np.argsort(scores)[::-1][:num_features]
#     return topN_indices
# 
# def vectorize_and_filter(texts, vectorizer, topN_indices=None):
#     vectors = vectorizer.transform(texts)
#     
#     if topN_indices is not None:
#         filtered_vectors = vectors[:, topN_indices]
#     else:
#         filtered_vectors = vectors
#     
#     feature_names = np.array(vectorizer.get_feature_names_out())[topN_indices]
#     filtered_df = pd.DataFrame(filtered_vectors.toarray(), columns=feature_names)
#     
#     return filtered_df
#
# vectorizer = initialize_vectorizer()
# 
# X_train_vectors = vectorizer.fit_transform(X_train['text'])
# topN_indices = filter_top_words(X_train_vectors, 20)
# 

# df_train_filtered = vectorize_and_filter(X_train['text'], vectorizer, topN_indices)
# 

# df_test_filtered = vectorize_and_filter(X_test['text'], vectorizer, topN_indices)
# 


# X_train.reset_index(drop=True, inplace=True)
# check_X_train.reset_index(drop=True, inplace=True)
# check_X_train = pd.merge(check_X_train, df_train_filtered,left_index=True, right_index=True)
# 
# X_test.reset_index(drop=True, inplace=True)
# check_X_test.reset_index(drop=True, inplace=True)
# check_X_test = pd.merge(check_X_test, df_test_filtered,left_index=True, right_index=True)
# 
# =============================================================================






##---------------------------------------DecisionTreeClassifier-----------------------------------------------------####
Decision_Tree_Model = DecisionTreeClassifier(criterion='entropy',max_depth=4)

CV_Scores = cross_val_score(Decision_Tree_Model, X_train, y_train, cv=10, scoring='roc_auc')
Decision_Tree_Model.fit(X_train, y_train)
print(f"Average AUC-ROC score from CV: {np.mean(CV_Scores):.3f}")
print(f"AUC-ROC training score: {roc_auc_score(y_train, Decision_Tree_Model.predict_proba(X_train)[:, 1]):.3f}")

param_grid = {'max_depth': [4,5,6], 'criterion': ['entropy'],
              'min_samples_leaf':[2,3,4,5,6,8,10,12], 'min_samples_split': [8,10,12],'ccp_alpha' : [0.01,0.015,0.02,0.025,0.03]}

# DecisionTreeClassifier
decision_tree_model  = DecisionTreeClassifier(criterion='entropy',random_state=42)

grid_search = GridSearchCV(estimator=decision_tree_model , param_grid=param_grid, cv=10, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)


#Hyper parameters
print("Best parameters found: ", grid_search.best_params_)
print("Best AUC-ROC score from CV: ", grid_search.best_score_)


#plot tree only on the top 3 layers
plt.figure(figsize=(16, 16))
plot_tree(grid_search.best_estimator_, filled=True, max_depth=3, feature_names=X_train.columns.tolist(), class_names=['0', '1'], fontsize=10)
plt.show()


optimized_model = grid_search.best_estimator_
y_train_pred_proba = optimized_model.predict_proba(X_train)[:, 1]
y_test_pred_proba = optimized_model.predict_proba(X_test)[:, 1]

auc_roc_train = roc_auc_score(y_train, y_train_pred_proba)
auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)

#print result
print(f"AUC-ROC training score (optimized): {auc_roc_train:.3f}")
print(f"AUC-ROC validation score: {auc_roc_test:.3f}")

#feature importances plot
importances = optimized_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_train.columns[indices]
df_importances = pd.DataFrame({'feature': features, 'importances': importances[indices]})
print(df_importances)

# Plot feature importances
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
plt.title("Top 10 Features importance Decision Tree Classifier", fontsize=18)
sns.barplot(x=importances[indices][:10], y=features[:10], palette="deep")
plt.xlabel("Relative Importance", fontsize=16)
plt.show()



def grid_search_cv(model, param_grid):
    comb = 1
    for list_ in param_grid.values():
        comb *= len(list_)
    print(comb)
    
##----------------------------------ANN---------------------------##
model_ANN = MLPClassifier(random_state=1, max_iter=1000, activation='logistic', learning_rate_init=0.01)
model_ANN.fit(X_train, y_train)

#clac roc
y_train_proba = model_ANN.predict_proba(X_train)[:, 1]
y_test_proba = model_ANN.predict_proba(X_test)[:, 1]
train_auc_roc = roc_auc_score(y_train, y_train_proba)
test_auc_roc = roc_auc_score(y_test, y_test_proba)

print(f"Train AUC-ROC score: {train_auc_roc:.3f}")
print(f"Train Accuracy score: {model_ANN.score(X_train, y_train):.3f}")
print(f"Test AUC-ROC score: {test_auc_roc:.3f}")
print(f"Test Accuracy score: {model_ANN.score(X_test, y_test):.3f}")

print(confusion_matrix(y_true=y_test, y_pred=model_ANN.predict(X_test)))

#ROC
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % test_auc_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#best model ANN
param_distributions = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)], 
    'activation': ['logistic'],  
    'learning_rate_init': [0.001, 0.0001],  
    'alpha': [0.0001, 0.001, 0.01]} 

mlp_base = MLPClassifier(max_iter=1000, random_state=1, early_stopping=True, validation_fraction=0.1)

#  RandomizedSearchCV
random_search = RandomizedSearchCV(mlp_base, param_distributions, n_iter=10, cv=5, scoring='roc_auc', random_state=1)

random_search.fit(X_train, y_train)
 
#Best model
best_model_mlp = random_search.best_estimator_
print("Best parameters found:", random_search.best_params_)

#calc auc-roc
train_auc_roc = roc_auc_score(y_train, best_model_mlp.predict_proba(X_train)[:, 1])
test_auc_roc = roc_auc_score(y_test, best_model_mlp.predict_proba(X_test)[:, 1])

print(f"Train AUC-ROC score: {train_auc_roc:.3f}")
print(f"Test AUC-ROC score: {test_auc_roc:.3f}")

print("Confusion matrix:")
print(confusion_matrix(y_test, best_model_mlp.predict(X_test)))

#ROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, best_model_mlp.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {test_auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


##---------------------------------------imporvment-----------------------------------------------------####
X_trainImprove= X_train
X_trainImprove.drop('number_of_new_followers', axis=1, inplace=True)
X_trainImprove.drop('number_of_new_follows', axis=1, inplace=True)


X_testImprove=X_test
X_testImprove.drop('number_of_new_followers', axis=1, inplace=True)
X_testImprove.drop('number_of_new_follows', axis=1, inplace=True)


Decision_Tree_Model_imp = DecisionTreeClassifier(criterion='entropy',max_depth=20)

CV_Scores = cross_val_score(Decision_Tree_Model_imp, X_trainImprove, y_train, cv=10, scoring='roc_auc')
Decision_Tree_Model_imp.fit(X_trainImprove, y_train)
print(f"Average AUC-ROC score from CV: {np.mean(CV_Scores):.3f}")
print(f"AUC-ROC training score: {roc_auc_score(y_train, Decision_Tree_Model_imp.predict_proba(X_trainImprove)[:, 1]):.3f}")

param_grid = {'max_depth': [5,6,7,8,9], 'criterion': ['entropy','gini'],
                  'min_samples_leaf':[2,3,4,5,6,8,10], 'min_samples_split': [2,3,4,5,6,7,8,10,],'ccp_alpha' : [0.01,0.015,0.02]}

#  DecisionTreeClassifier
decision_tree_model  = DecisionTreeClassifier(criterion='entropy',random_state=42)

grid_search = GridSearchCV(estimator=decision_tree_model , param_grid=param_grid, cv=10, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)


#Hyper parameters
print("Best parameters found: ", grid_search.best_params_)
print("Best AUC-ROC score from CV: ", grid_search.best_score_)


#plot tree only on the top 3 layers
plt.figure(figsize=(16, 16))
plot_tree(grid_search.best_estimator_, filled=True, max_depth=3, feature_names=X_trainImprove.columns.tolist(), class_names=['0', '1'], fontsize=10)
plt.show()


optimized_model = grid_search.best_estimator_
y_train_pred_proba = optimized_model.predict_proba(X_trainImprove)[:, 1]
y_test_pred_proba = optimized_model.predict_proba(X_testImprove)[:, 1]

auc_roc_train = roc_auc_score(y_train, y_train_pred_proba)
auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)

    #print result
print(f"AUC-ROC training score (optimized): {auc_roc_train:.3f}")
print(f"AUC-ROC validation score: {auc_roc_test:.3f}")

    #feature importances plot
importances = optimized_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_trainImprove.columns[indices]
df_importances = pd.DataFrame({'feature': features, 'importances': importances[indices]})
print(df_importances)

    # Plot feature importances
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
plt.title("Top 10 Features importance Decision Tree Classifier", fontsize=18)
sns.barplot(x=importances[indices][:10], y=features[:10], palette="deep")
plt.xlabel("Relative Importance", fontsize=16)
plt.show()


###RANDOM FOREST
# model_rf = RandomForestClassifier(random_state=42)
# model_rf.fit(X_train, y_train)
# 
# y_train_pred = model_rf.predict_proba(X_train)[:, 1]
# y_test_pred = model_rf.predict_proba(X_test)[:, 1]
# 
# train_auc_roc = roc_auc_score(y_train, y_train_pred)
# test_auc_roc = roc_auc_score(y_test, y_test_pred)
# 
# print(f"Train AUC-ROC score (without tuning): {train_auc_roc:.3f}")
# print(f"Test AUC-ROC score (without tuning): {test_auc_roc:.3f}")
# 

# param_distributions = {
#     'n_estimators': [10, 50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
# 

# random_search = RandomizedSearchCV(
#     RandomForestClassifier(random_state=42),
#     param_distributions=param_distributions,
#     n_iter=10,  
#     cv=3,
#     scoring='roc_auc',
#     random_state=42
# )
# 
# random_search.fit(X_train, y_train)
# 
# best_model = random_search.best_estimator_
# 
# y_train_best_pred = best_model.predict_proba(X_train)[:, 1]
# y_test_best_pred = best_model.predict_proba(X_test)[:, 1]
# 
# train_best_auc_roc = roc_auc_score(y_train, y_train_best_pred)
# test_best_auc_roc = roc_auc_score(y_test, y_test_best_pred)
# 
# print(f"Train AUC-ROC score (with tuning): {train_best_auc_roc:.3f}")
# print(f"Test AUC-ROC score (with tuning): {test_best_auc_roc:.3f}")
# print(f"Best hyperparameters: {random_search.best_params_}")
#
# =============================================================================

#####PREDICTIONS####
X_process_test_set = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\X_test_process.pkl')
X_process_test_set.drop('sentiment', axis=1, inplace=True)  
predictions = optimized_model.predict(X_process_test_set)
G18_ytest = pd.DataFrame(predictions, columns=['sentiment'])
G18_ytest['sentiment'] = G18_ytest['sentiment'].apply(lambda x: 'positive' if x == 1 else 'negative')
G18_ytest.to_pickle('G18_ytest.pkl')
