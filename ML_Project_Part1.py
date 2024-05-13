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


trainSet = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\XY_train.pkl')
checkforduplicate = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\XY_train.pkl')

y_test_sample = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\sample_y_test.pkl')
#trainSet = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\X_test.pkl')
#trainSet['sentiment'] = np.random.choice(['F', 'M'], size=len(trainSet))
#checkforduplicate = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\X_test.pkl')
#checkforduplicate['sentiment'] = np.random.choice(['F', 'M'], size=len(checkforduplicate))

checkforduplicate.drop('textID', axis=1, inplace=True)  

Copy_train_Set= copy.deepcopy(trainSet) #copy the trainSet
for col in Copy_train_Set.columns: #conversion every col for string
    Copy_train_Set[col] = Copy_train_Set[col].apply(str)


########### Pre-Processing + Feature_Extraction ##########
#Check for Duplicate Samples:
duplicate_rows = checkforduplicate[Copy_train_Set.duplicated(keep=False)] #Check for duplicate samples
if duplicate_rows.shape[0] > 0:
    print("There are identical duplicate rows!")
else:
    print("No identical duplicates found")
    

#Set all null values with the same word- "Unknown" for Data consistency
def replace_nulls(df, column, new_value):

  mask = (df[column] == "None") | (df[column] == "nan")  
  df.loc[mask, column] = new_value
replace_nulls(Copy_train_Set, "email", "Unknown")
replace_nulls(Copy_train_Set , "gender", "Unknown") 
replace_nulls(Copy_train_Set, "email_verified", "Unknown")
replace_nulls(Copy_train_Set, "blue_tick", "Unknown")
replace_nulls(Copy_train_Set, "embedded_content", "Unknown")
replace_nulls(Copy_train_Set, "platform", "Unknown")


#new feature -> seniority
#add col seniority (from account_creation_date -> seniority)
Copy_train_Set['account_creation_date'] = pd.to_datetime(Copy_train_Set['account_creation_date'])
Copy_train_Set['account_creation_date'] = Copy_train_Set['account_creation_date'].dt.date
today = datetime.date.today()  
delta = Copy_train_Set['account_creation_date'].apply(lambda date: (today - date).days / 365)
Copy_train_Set['seniority'] = delta

#aggregate different dates at the col
print(Copy_train_Set["previous_messages_dates"].apply(type).unique()) 

#Count number of differents dates
def count_expressions(text):
  if text == []:
     return 0
  return len(re.findall("'[^']*'", text))
#Call for count_expressions FUNC
Copy_train_Set["number_of_previous_messages"] = Copy_train_Set["previous_messages_dates"].apply(count_expressions) #add new col for number_of_previous_messages
Copy_train_Set["number_of_new_followers"] = Copy_train_Set["date_of_new_follower"].apply(count_expressions) #add new col for number of new followers
Copy_train_Set["number_of_new_follows"] = Copy_train_Set["date_of_new_follow"].apply(count_expressions) #add new col for number of new followers

#get the earliest date(year) from all the cell
def get_earliest_date(text):

  if text == []:
    return "Unknown"

  dates_strings = re.findall("'(.+?)'", text)
  
#Check for dates
  if len(dates_strings) > 0:

 #Conversion for dates
    dates = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dates_strings]
    earliest_date = min(dates)
    return earliest_date.year
  
  else: 
    return "Unknown"


#new features from dates columns (Ealier)
Copy_train_Set["Earlier_date_of_follower"] = Copy_train_Set["date_of_new_follower"].apply(get_earliest_date)
Copy_train_Set["Earlier_date_of_follow"] = Copy_train_Set["date_of_new_follow"].apply(get_earliest_date)

#Get the lastest date(year) from all the cell
def get_lastest_date(text):
    if text == []:
      return "Unknown"

    dates_strings = re.findall("'(.+?)'", text)
    
  #Check for dates
    if len(dates_strings) > 0:

   #conversion for dates
      dates = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dates_strings]
      earliest_date = max(dates)
      return earliest_date.year
    
    else: 
      return "Unknown"
  
#new features from date colmuns (Latest)
Copy_train_Set["Latest_date_of_follower"] = Copy_train_Set["date_of_new_follower"].apply(get_lastest_date)
Copy_train_Set["Latest_date_of_follow"] = Copy_train_Set["date_of_new_follow"].apply(get_lastest_date)

#Get the email's company name
def get_company(email):
  if email == "Unknown":
    return "Unknown"
  
  parts = re.split("@", email)  
  
  if len(parts) > 1:
    company = parts[1].split(".")[0]
    return company

  return "Unknown"

#create new col for email_company
Copy_train_Set["email_company"] = Copy_train_Set["email"].apply(get_company)

###message_year####
years = Copy_train_Set['message_date'].apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").year)

# Added the year as a new column
Copy_train_Set['message_year'] = years



#use textBlob for sentiment prediction ###check if it is fine for that
#def get_sentiment(text):
#    blob = TextBlob(text)
#    return blob.sentiment.polarity, blob.sentiment.subjectivity

#sentiment_scores = Copy_train_Set["text"].apply(get_sentiment)

#seperate the columns 
#polarity_column = [score[0] for score in sentiment_scores]  
#subjectivity_column = [score[1] for score in sentiment_scores]

#add two columns
#Copy_train_Set["polarity by TextBlob"] = polarity_column #between -1 to 1 -> more positive when get higer
#Copy_train_Set["subjectivity by TextBlob"] = subjectivity_column #objective/subjective -> 0 is more objective and 1 more subjective


#some text  features
def get_text_features(text):

  num_words = len(text.split())
  num_sentences = text.count('.') + 1
  num_chars = len(text)
  avg_word_length = sum(len(word) for word in text.split()) / num_words
  
  return {
    "length": num_chars, 
    "words": num_words,
    "sentences": num_sentences,
    "avg_word_length": avg_word_length
  }

text_features = Copy_train_Set["text"].apply(get_text_features)

char_count_col = [x["length"] for x in text_features]  
Copy_train_Set["char_count"] = char_count_col #count chars

word_count_col = [x["words"] for x in text_features]
Copy_train_Set["word_count"] = word_count_col #count words

sentence_count_col = [x["sentences"] for x in text_features]
Copy_train_Set["sentence_count"] = sentence_count_col #count sentences

avg_word_length_col = [x["avg_word_length"] for x in text_features]
Copy_train_Set["avg_word_length"] = avg_word_length_col #check for avg of words len

#check for hour of message datetime
Copy_train_Set["message_date"] = pd.to_datetime(Copy_train_Set["message_date"])

#discrization for message_time from message_date
def get_time_of_day(datetime):
  hour = datetime.hour
  if hour >= 0 and hour < 8:
    return "morning"
  elif hour >= 8 and hour < 20:  
    return "afternoon"
  else:
    return "evening"

#Add new col- message_time
time_of_day = Copy_train_Set["message_date"].apply(get_time_of_day)
Copy_train_Set["message_time"] = time_of_day


### export for pkl file ###
#updated_df = Copy_train_Set  



# שמירה לקובץ #PKL
with open('Copy_train_Set.pkl', 'wb') as file:  
    pickle.dump(Copy_train_Set, file)
    
#print('Dataset exported to PKL file!')
#with open('updated_dataset.pkl', 'rb') as file:
#    loaded_df = pickle.load(file)

# =============================================================================
# ################TF-IDF########################
# #Load_texts
# texts = Copy_train_Set['text']
# def vectorize_and_filter(texts, num_features):
#  #vectorizer
#     vectorizer = TfidfVectorizer(max_df=0.1)
#     vectors = vectorizer.fit_transform(texts)
# #filter important words
#     indices = filter_top_words(vectors, num_features)
#     filtered_vectors = vectors[:, indices]
# #series of data
#     filtered_df = vectors_to_df(filtered_vectors, vectorizer, indices)
#     return filtered_df
#     df = vectorize_and_filter(Copy_train_Set['text'], 25)
# 
# def filter_top_words(vectors, N):
#     scores = np.asarray(vectors.sum(axis=0)).flatten()
#     topN = np.argsort(scores)[::-1][:N]
#     return topN
# 
# 
# def vectors_to_df(vectors, vectorizer, indices):
#     feature_names = vectorizer.get_feature_names_out()
#     words = [feature_names[i] for i in indices]
#     return pd.DataFrame(vectors.toarray(), columns=words)
# 
# 
# df = vectorize_and_filter(Copy_train_Set['text'], 25)
# 
# ####The combination of features from TF-IDF####
# Copy_train_Set = pd.merge(Copy_train_Set, df, left_index=True, right_index=True)
# =============================================================================



#Fill values with a priori probability
#sampeling to gender for missing values
p_female = 0.4
p_male = 0.47 
p_uknown = 0.13

for idx, row in Copy_train_Set.iterrows():

  if pd.isnull(row['gender']) or row['gender'] == 'Unknown':
  
    gender = None
    
    while gender is None:

      rnd = np.random.random()
      if rnd <= p_female: 
        gender = 'F'
      elif rnd > p_female and rnd < 0.87:
        gender = 'M'  
      else:
        rnd = np.random.random()
        
      Copy_train_Set.at[idx,'gender'] = gender
      


#sampling to email_verified for missing values
p_true = 0.46
p_false = 0.45
p_uknown = 0.09

for idx, row in Copy_train_Set.iterrows():

  if pd.isnull(row['email_verified']) or row['email_verified'] == 'Unknown':
  
    email_verified = None
    
    while email_verified is None:

      rnd = np.random.random()
      if rnd <= p_true: 
        email_verified = 'True'
      elif rnd > p_true and rnd < 0.91:
        email_verified = 'False'  
      else:
        rnd = np.random.random()
        
      Copy_train_Set.at[idx,'email_verified'] = email_verified
      


#sampling to blue_tick for missing values
p_true = 0.63
p_false = 0.25
p_uknown = 0.12

      
for idx, row in Copy_train_Set.iterrows():

  if pd.isnull(row['blue_tick']) or row['blue_tick'] == 'Unknown':
  
    blue_tick = None
    
    while blue_tick is None:

      rnd = np.random.random()
      if rnd <= p_true: 
        blue_tick = 'True'
      elif rnd > p_true and rnd < 0.88:
        blue_tick = 'False'  
      else:
        rnd = np.random.random()
        
      Copy_train_Set.at[idx,'blue_tick'] = blue_tick      

#sampling to embedded_content for missing values
      
p_false = 0.55
p_mp4 = 0.16       
p_link = 0.12   
p_jpeg = 0.09
p_unkown = 0.08
for idx, row in Copy_train_Set.iterrows():

  if pd.isnull(row['embedded_content']) or row['embedded_content'] == 'Unknown':
  
    embedded_content = None
    
    while embedded_content is None:

      rnd = np.random.random()
      if rnd <= p_false: 
        embedded_content = 'False'
      elif rnd > p_false and rnd < 0.71:
        embedded_content = 'mp4' 
      elif rnd > 0.71 and rnd < 0.83:
        embedded_content = 'link'
      elif rnd > 0.83 and rnd < 0.92:
       embedded_content= 'jpeg'
      else: rnd = np.random.random()
      Copy_train_Set.at[idx,'embedded_content'] = embedded_content     


#sampling to platform for missing values
p_facebook = 0.23
p_tiktok = 0.18
p_x = 0.16
p_telegram = 0.15
p_instgram = 0.14
p_whatsapp = 0.08
p_Unkown = 0.06


for idx, row in Copy_train_Set.iterrows():

  if pd.isnull(row['platform']) or row['platform'] == 'Unknown':
  
    platform = None
    
    while platform is None:

      rnd = np.random.random()
      if rnd <= p_facebook: 
        platform = 'facebook'
      elif rnd > p_facebook and rnd < 0.41:
        platform = 'tiktok' 
      elif rnd > 0.41 and rnd < 0.57:
        platform = 'x'
      elif rnd > 0.57 and rnd < 0.72:
       platform= 'telegram'
      elif rnd > 0.72 and rnd < 0.86:
        platform= 'instagram'
      elif rnd > 0.86 and rnd < 0.94:
          platform= 'whatsapp'
      else: rnd = np.random.random()
      Copy_train_Set.at[idx,'platform'] = platform 

#sampeling emails for missing values
#only for email_company rows with unknown
unknown_rows = Copy_train_Set[Copy_train_Set['email_company'] == 'Unknown']  

#Random sampling from existing values
sample_emails = Copy_train_Set[Copy_train_Set['email_company'] != 'Unknown'].sample(len(unknown_rows))  

#fill random sampels
for email, idx in zip(sample_emails, unknown_rows.index):
    Copy_train_Set.loc[idx, 'email_company'] = email
    


#convert to Binary
mapping2gender = {'M': 1, 'F': 0}
Copy_train_Set['gender'] = Copy_train_Set['gender'].map(mapping2gender)

mapping2email_verified = {'True': 1, 'False' : 0}
Copy_train_Set['email_verified'] = Copy_train_Set['email_verified'].map(mapping2email_verified)

mapping2blue_tick = {'True' : 1, 'False' : 0}
Copy_train_Set['blue_tick'] = Copy_train_Set['blue_tick'].map(mapping2blue_tick)

mapping2sentiment = {'positive' : 1, 'negative' : 0}
Copy_train_Set['sentiment'] = Copy_train_Set['sentiment'].map(mapping2sentiment)



#FOR HEAT MAP - PERSON CORR FOR ALL THE NUMERIC VALUES
Copy_train_Set['Earlier_date_of_follower'] = pd.to_numeric(Copy_train_Set['Earlier_date_of_follower'], errors='coerce')
Copy_train_Set['Earlier_date_of_follow'] = pd.to_numeric(Copy_train_Set['Earlier_date_of_follow'], errors='coerce')
Copy_train_Set['Latest_date_of_follower'] = pd.to_numeric(Copy_train_Set['Latest_date_of_follower'], errors='coerce')
Copy_train_Set['Latest_date_of_follow'] = pd.to_numeric(Copy_train_Set['Latest_date_of_follow'], errors='coerce')

#calculate MEAN FOR any of variable
mean_Earlier_date_of_follower = round(Copy_train_Set['Earlier_date_of_follower'].mean())
mean_Earlier_date_of_follow = round(Copy_train_Set['Earlier_date_of_follow'].mean())
mean_Latest_date_of_follower = round(Copy_train_Set['Latest_date_of_follower'].mean())
mean_Latest_date_of_follow = round(Copy_train_Set['Latest_date_of_follow'].mean())

#Convert Unknown to null
Copy_train_Set['Earlier_date_of_follower'] = Copy_train_Set['Earlier_date_of_follower'].replace("Unknown", np.nan)
Copy_train_Set['Earlier_date_of_follow'] = Copy_train_Set['Earlier_date_of_follow'].replace("Unknown", np.nan)
Copy_train_Set['Latest_date_of_follower'] = Copy_train_Set['Latest_date_of_follower'].replace("Unknown", np.nan)
Copy_train_Set['Latest_date_of_follow'] = Copy_train_Set['Latest_date_of_follow'].replace("Unknown", np.nan)

#Fillin missing values with the mean in each column
Copy_train_Set['Earlier_date_of_follower'] = Copy_train_Set['Earlier_date_of_follower'].fillna(mean_Earlier_date_of_follower)
Copy_train_Set['Earlier_date_of_follow'] = Copy_train_Set['Earlier_date_of_follow'].fillna(mean_Earlier_date_of_follow)
Copy_train_Set['Latest_date_of_follower'] = Copy_train_Set['Latest_date_of_follower'].fillna(mean_Latest_date_of_follower)
Copy_train_Set['Latest_date_of_follow'] = Copy_train_Set['Latest_date_of_follow'].fillna(mean_Latest_date_of_follow)

#Numeric variables for analysis
columns = ['seniority', 'number_of_previous_messages', 'number_of_new_followers', 
           'number_of_new_follows', 'char_count', 'word_count', 'sentence_count',
           'avg_word_length','Earlier_date_of_follower','Earlier_date_of_follow','Latest_date_of_follower','Latest_date_of_follow'
           ,'blue_tick','email_verified','gender','sentiment']

          
#person corr     
plt.figure(figsize=(14,12))      
heatmap = sns.heatmap(Copy_train_Set[columns].corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
heatmap.set_title('Correlation Heatmap')
plt.show()


################################################### Feature Representation ############################################
# One_hot_encoding for embedded_content
embedded_content_dummies = pd.get_dummies(Copy_train_Set['embedded_content'], prefix='embedded_content')
platform_dummies = pd.get_dummies(Copy_train_Set['platform'], prefix='platform')
message_time_dummies = pd.get_dummies(Copy_train_Set['message_time'], prefix='message_time')
# Add to Copy_train_Set
Copy_train_Set = pd.concat([Copy_train_Set, embedded_content_dummies], axis=1)
Copy_train_Set = pd.concat([Copy_train_Set, platform_dummies], axis=1)
Copy_train_Set = pd.concat([Copy_train_Set, message_time_dummies], axis=1)


# One-hot encoding for message_time


# Adding the one-hot encoded columns to Copy_train_Set


# Convert to binary
Copy_train_Set['embedded_content_mp4'] = Copy_train_Set['embedded_content_mp4'].replace({True: 1, False: 0})
Copy_train_Set['embedded_content_link'] = Copy_train_Set['embedded_content_link'].replace({True: 1, False: 0})
Copy_train_Set['embedded_content_False'] = Copy_train_Set['embedded_content_False'].replace({True: 1, False: 0})
Copy_train_Set['embedded_content_jpeg'] = Copy_train_Set['embedded_content_jpeg'].replace({True: 1, False: 0})

Copy_train_Set['platform_whatsapp'] = Copy_train_Set['platform_whatsapp'].replace({True: 1, False: 0})
Copy_train_Set['platform_telegram'] = Copy_train_Set['platform_telegram'].replace({True: 1, False: 0})
Copy_train_Set['platform_x'] = Copy_train_Set['platform_x'].replace({True: 1, False: 0})
Copy_train_Set['platform_instagram'] = Copy_train_Set['platform_instagram'].replace({True: 1, False: 0})
Copy_train_Set['platform_facebook'] = Copy_train_Set['platform_facebook'].replace({True: 1, False: 0})
Copy_train_Set['platform_tiktok'] = Copy_train_Set['platform_tiktok'].replace({True: 1, False: 0})

Copy_train_Set['message_time_morning'] = Copy_train_Set['message_time_morning'].replace({True: 1, False: 0})
Copy_train_Set['message_time_afternoon'] = Copy_train_Set['message_time_afternoon'].replace({True: 1, False: 0})
Copy_train_Set['message_time_evening'] = Copy_train_Set['message_time_evening'].replace({True: 1, False: 0})


# Frequency Encoding for Email- too many of categories
email_freq = Copy_train_Set['email_company'].value_counts(normalize=True)
#replace the values for numeric
Copy_train_Set['email_company'] = Copy_train_Set['email_company'].map(email_freq)



#Normalization of variables by min-max scaling
scaler = MinMaxScaler()

#pick col for scaler
columns_to_scale = ['seniority', 'number_of_previous_messages', 'number_of_new_followers', 
                    'number_of_new_follows', 'char_count', 'word_count', 'sentence_count',
                    'avg_word_length','Earlier_date_of_follower','Earlier_date_of_follow',
                    'Latest_date_of_follower','Latest_date_of_follow','message_year']
#Normalization
Copy_train_Set[columns_to_scale] = scaler.fit_transform(Copy_train_Set[columns_to_scale])


# =============================================================================
# ####FISCHER SCORE####
# #Number of features for fischer score
# k_features = 50
# 
# #selector for fischer score
# selector = SelectKBest(score_func=f_classif, k=k_features)
# 
# # Assembling the data with the filtered columns
# def select_k_best_features(data, target_variable, k_features):
#     # בניית מערך המאפיינים העצמאיים והמשתנה התלוי
#     X = data.drop(columns=['textID', 'text', 'message_date', 'account_creation_date',
#                            'previous_messages_dates', 'date_of_new_follower',
#                            'date_of_new_follow', 'email', 'embedded_content', 'platform', 'message_time'])
#     y = data[target_variable]
# 
# # Selecting the appropriate feature
#     X_selected = selector.fit_transform(X, y)
# 
#     return X_selected
# 
# 
# X_selected = select_k_best_features(Copy_train_Set, 'sentiment', k_features)
# 
# feature_names = Copy_train_Set.drop(columns=['textID', 'text', 'message_date', 'account_creation_date',
#                                                       'previous_messages_dates', 'date_of_new_follower',
#                                                       'date_of_new_follow', 'email', 'embedded_content', 'platform', 'message_time']).columns
# 
# 
# #show fischer score result
# def plot_feature_scores(feature_scores, feature_names):
# # Sort the column names according to the Fisher Review values
#     sorted_features = [f for _, f in sorted(zip(feature_scores, feature_names), reverse=True)]
#     sorted_scores = sorted(feature_scores, reverse=True)
# 
#     plt.figure(figsize=(12, 20)) 
#     bars = plt.barh(range(len(sorted_features)), sorted_scores, color='skyblue', height=0.7, tick_label=sorted_features)  # מייצג ברים אופקיים
#     plt.xlabel('Fisher Score')  
#     plt.ylabel('Feature Name') 
#     plt.title('Fisher Scores of Selected Features')  
#     for bar, score in zip(bars, sorted_scores):
#         plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{score:.2f}', ha='left', va='center', fontsize=10)  # הוספת המספר על הבר
#     plt.ylim(-0.5, len(sorted_features) - 0.5)  
#     plt.subplots_adjust(hspace=0.5)  
#     plt.tight_layout()  
#     plt.show()
# 
# #plot for fischer score
# plot_feature_scores(selector.scores_, feature_names)
# =============================================================================


########## FINAL DATA SET FOR TRAINING ##########

FinalTrainSet=copy.deepcopy(Copy_train_Set)
FinalTrainSet = FinalTrainSet.drop(columns=['textID', 'text', 'message_date', 'account_creation_date',
                                                      'previous_messages_dates', 'date_of_new_follower',
                                                      'date_of_new_follow', 'email', 'embedded_content', 'platform', 'message_time'])


FinalTrainSet.to_pickle('FinalTrainSet2.pkl')






