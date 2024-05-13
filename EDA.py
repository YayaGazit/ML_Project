# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:56:56 2024

@author: ygazi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


trainSetUpdated = pd.read_pickle(r'C:\Users\ygazi\Desktop\יאיא\בן גוריון\לימוד מכונה\XY_train_Updated.pkl')


# Calculate sentiment counts
#Count by positive sentiment and negative sentiment
pos_count = trainSetUpdated[trainSetUpdated["sentiment"] == "positive"].shape[0]  
neg_count = trainSetUpdated[trainSetUpdated["sentiment"] == "negative"].shape[0]

#values for the graph
labels = ['Positive', 'Negative']   
sizes = [pos_count, neg_count]
explode = (0, 0)  
colors = ['lightblue', 'orange']    

#PIE CHART to check Balance in the target variable
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%') 

plt.axis('equal')
plt.title("Sentiments Percentage") 
plt.tight_layout()
plt.show()
plt.show()



#explore text  - for positive text. The 15 words that repeat themselves the most
def preprocess(text, stop_words):
  words = [word for word in text.split() if word not in stop_words]  
  words = [word.strip() for word in words]
  return words


text_series = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive']['text']
all_words = []
stop_words = ['this', 'and', 'but', 'the','is','to','I','a','my','and','i','you','for','in','of',

'it','so','','on','that','with','be',"I'm",'was','at','-','are','me','I`m']

for text in text_series:
  words = preprocess(text, stop_words)
  all_words.extend(words)

total_words = len(all_words)
word_counts = Counter(all_words)  

top_15_words = word_counts.most_common(15)
print("Top 15 words in positive sentiment:")  
for word, count in top_15_words:
  percentage = (count / total_words) * 100
  
  print(f"{word}: {percentage:.2f}%")
  
#for negative text.  The 15 words that repeat themselves the most
text_series_neg = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']['text']
for neg in text_series_neg:
  words = preprocess(neg, stop_words)  
  all_words.extend(words)

total_words = len(all_words)
word_counts = Counter(all_words)
   
top_15_words = word_counts.most_common(15)

print("Top 15 words in negative sentiment:")  
for word, count in top_15_words:
  percentage = (count / total_words) * 100
  print(f"{word}: {percentage:.2f}%")


#Seniority data by sentiment groups 
positive_seniority = trainSetUpdated[trainSetUpdated['sentiment']=="positive"]['seniority']
negative_seniority = trainSetUpdated[trainSetUpdated['sentiment']=="negative"]['seniority']

#BOXPLOT
ax = sns.boxplot(x="sentiment", y="seniority", data=trainSetUpdated)
ax.set_title("Seniority distrubtion by sentiment")
ax.set_xticklabels(['Positive','Negative'])

plt.show()

#HISTOGRAMA
plt.hist(trainSetUpdated['seniority'], bins=50)
plt.title('Distribution of Number of seniority')
plt.xlabel('Number of seniority')
plt.ylabel('Count')
plt.yticks(np.arange(0, 750, 50)) 
plt.xticks(np.arange(7,13,1))
plt.figure(figsize=(4,5))
plt.show()



#BOXPLOT
ax = sns.boxplot(x="sentiment", y="number_of_previous_messages", data=trainSetUpdated)

ax.set_title("Number of Previous Messages by Sentiment")
ax.set(xlabel="Sentiment", ylabel="Number of Previous Messages")

plt.show()



#HISTOGRAMA
plt.hist(trainSetUpdated['number_of_previous_messages'], bins=50)
plt.title('Distribution of Number of Previous Messages')
plt.xlabel('Number of Previous Messages')
plt.ylabel('Count')
plt.yticks(np.arange(0, 800, 100)) 
plt.xticks(np.arange(0,50,5))
plt.figure(figsize=(15,8))
plt.show()

#BOXPLOT
ax = sns.boxplot(x="sentiment", y="number_of_new_follows", data=trainSetUpdated) 
ax.set_title("Number of New Follows by Sentiment")
ax.set(xlabel="Sentiment", ylabel="Number of New Follows")
plt.show()

#HISTOGRAMA
plt.hist(trainSetUpdated['number_of_new_follows'], bins=50)
plt.title('Distribution of number_of_new_follows')
plt.xlabel('number_of_new_follows')
plt.ylabel('Count')
plt.yticks(np.arange(0, 800, 100)) 
plt.xticks(np.arange(0,50,5))
plt.figure(figsize=(15,8))
plt.show()



#BOXPLOT
ax = sns.boxplot(x="sentiment", y="number_of_new_followers", data=trainSetUpdated) 
ax.set_title("Number of New Followers by Sentiment")
ax.set(xlabel="Sentiment", ylabel="Number of New Followers")
plt.show()

#HISTOGRAMA
plt.hist(trainSetUpdated['number_of_new_followers'], bins=50)
plt.title('Distribution of number_of_new_followers')
plt.xlabel('number_of_new_followers')
plt.ylabel('Count')
plt.yticks(np.arange(0, 800, 100)) 
plt.xticks(np.arange(0,50,5))
plt.figure(figsize=(15,8))
plt.show()


#BOXPLOT
ax = sns.boxplot(x="sentiment", y="char_count", data=trainSetUpdated) 
ax.set_title("Number of char_count by Sentiment")
ax.set(xlabel="Sentiment", ylabel="Number of char_count")
plt.show()


#HISTOGRAMA
plt.hist(trainSetUpdated['char_count'], bins=30)
plt.title('Distribution of char_count')
plt.xlabel('char_count')
plt.ylabel('Count')
plt.yticks(np.arange(0, 800, 50)) 
plt.xticks(np.arange(0,150,10))
plt.show()


#BOXPLOT
ax = sns.boxplot(x="sentiment", y="word_count", data=trainSetUpdated) 
ax.set_title("Number of word_count by Sentiment")
ax.set(xlabel="Sentiment", ylabel="Number of word_count")
plt.show()

#HISTOGRAMA
plt.hist(trainSetUpdated['word_count'], bins=50)
plt.title('Distribution of word_count')
plt.xlabel('word_count')
plt.ylabel('Count')
plt.yticks(np.arange(0, 800, 100)) 
plt.xticks(np.arange(0,35,5))
plt.figure(figsize=(15,8))
plt.show()



#BOXPLOT
ax = sns.boxplot(x="sentiment", y="sentence_count", data=trainSetUpdated) 
ax.set_title("Number of sentence_count by Sentiment")
ax.set(xlabel="Sentiment", ylabel="Number of sentence_count")
plt.show()

#HISTOGRAMA
plt.hist(trainSetUpdated['sentence_count'], bins=50)
plt.title('Distribution of sentence_count')
plt.xlabel('sentence_count')
plt.ylabel('Count')
plt.yticks(np.arange(0, 7000, 500)) 
plt.xticks(np.arange(0,10,1))
plt.figure(figsize=(5,10))
plt.show()


#BOXPLOT
ax = sns.boxplot(x="sentiment", y="avg_word_length", data=trainSetUpdated) 
ax.set_title("Number of avg_word_length by Sentiment")
ax.set(xlabel="Sentiment", ylabel="Number of avg_word_length")
plt.show()

#HISTOGRAMA
plt.hist(trainSetUpdated['avg_word_length'], bins=30)
plt.title('Distribution of avg_word_length')
plt.xlabel('avg_word_length')
plt.ylabel('Count')
plt.yticks(np.arange(0, 7500, 500)) 
plt.xticks(np.arange(0,15,5))
plt.show()


# Count sentiments by platform
pos_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'].groupby('platform')['textID'].count()
neg_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative'].groupby('platform')['textID'].count()
total_by_platform = trainSetUpdated.groupby('platform')['textID'].count()

# Calculate percentages by sentiments
pos_percent = pos_count / total_by_platform * 100   
neg_percent = neg_count / total_by_platform * 100

# stacked bar plot 
sentiment_percent = pd.concat([pos_percent, neg_percent], axis=1)
sentiment_percent.columns = ['Positive', 'Negative'] 

ax = sentiment_percent.plot.bar(stacked=True,figsize=(9,6)) 
ax.set_xticklabels(['Unkown', 'facebook','instgram','telegram','tiktok','whatsapp','x'])
ax.set_ylabel("Percentage") 
plt.yticks(np.arange(0, 110, 10)) 
plt.yticks(np.arange(0, 110, 10), ['%d%%'% x for x in np.arange(0, 110, 10)])
plt.title("Sentiment Percentage per Platform", size=16)  
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.show()




#POSITIVE PIE CHART BY PLATFORM
pos_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'] 
pos_counts = pos_df['platform'].value_counts()
colors = ['lightblue', 'gold', 'lightgreen', 'purple', 'orange','silver','pink','green','red','grey','Turquoise','lightgrey']
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pos_counts, labels = pos_counts.index, autopct='%.1f%%',colors=colors)
ax1.set_title("Positive Sentiment by platform")


#NEGATIVE PIE CHART BY PLATFORM
neg_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']
neg_counts = neg_df['platform'].value_counts()
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.pie(neg_counts, labels = neg_counts.index, autopct='%.1f%%',colors=colors)
ax2.set_title("Negative Sentiment by platform")
plt.show()


#count values
counts = trainSetUpdated['platform'].value_counts()

#creath the graphes
plt.pie(counts, 
        labels=counts.index,  
        autopct='%1.1f%%',
        colors=['lightblue','pink','yellow','orange','lightgreen','silver','gold']
       )

plt.title('platform Percentage')
plt.axis('equal') 
plt.tight_layout()
plt.show() 



# Count sentiments by message_time
pos_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'].groupby('message_time')['textID'].count()
neg_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative'].groupby('message_time')['textID'].count()
total_by_platform = trainSetUpdated.groupby('message_time')['textID'].count()

# Calculate percentages
pos_percent = pos_count / total_by_platform * 100   
neg_percent = neg_count / total_by_platform * 100

#RELATIVE STACKED BAR
sentiment_percent = pd.concat([pos_percent, neg_percent], axis=1)
sentiment_percent.columns = ['Positive', 'Negative'] 

ax = sentiment_percent.plot.bar(stacked=True,figsize=(9,6)) 
ax.set_xticklabels(['morning', 'evening','afternoon'])
ax.set_ylabel("Percentage") 
plt.yticks(np.arange(0, 110, 10)) 
plt.yticks(np.arange(0, 110, 10), ['%d%%'% x for x in np.arange(0, 110, 10)])
plt.title("Sentiment Percentage per Message time", size=16)  
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.show()


#POSTIVE SENTIMENT by message_TIME PIE CHART
pos_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'] 
pos_counts = pos_df['message_time'].value_counts()
colors = ['lightblue', 'gold', 'lightgreen', 'purple', 'orange','silver','pink','green','red','grey','Turquoise','lightgrey']
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pos_counts, labels = pos_counts.index, autopct='%.1f%%',colors=colors)
ax1.set_title("Positive Sentiment by message_time")


#NEGATIVE SENTIMENT by message_TIME PIE CHART
neg_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']
neg_counts = neg_df['message_time'].value_counts()
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.pie(neg_counts, labels = neg_counts.index, autopct='%.1f%%',colors=colors)
ax2.set_title("Negative Sentiment by message_time")
plt.show()



# Count sentiments by email_company
pos_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'].groupby('email_company')['textID'].count()
neg_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative'].groupby('email_company')['textID'].count()
total_by_platform = trainSetUpdated.groupby('email_company')['textID'].count()
# Calculate percentages
pos_percent = pos_count / total_by_platform * 100   
neg_percent = neg_count / total_by_platform * 100

#RELATIVE STACKED BAR
sentiment_percent = pd.concat([pos_percent, neg_percent], axis=1)
sentiment_percent.columns = ['Positive', 'Negative'] 

ax = sentiment_percent.plot.bar(stacked=True,figsize=(9,6)) 
ax.set_ylabel("Percentage") 

plt.yticks(np.arange(0, 110, 10)) 
plt.yticks(np.arange(0, 110, 10), ['%d%%'% x for x in np.arange(0, 110, 10)])
plt.title("Sentiment Percentage per Email Company", size=16)  
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.show()


#POSITIVE PIE CHART by email_Company
pos_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'] 
pos_counts = pos_df['email_company'].value_counts()
pos_sorted_counts = pos_counts.sort_values(ascending=False)
pos_others = pos_sorted_counts[pos_sorted_counts < len(pos_df) * 0.04].sum()
pos_counts = pos_counts[pos_counts > len(pos_df) * 0.04] 
pos_counts['Others'] = pos_others
colors = ['lightblue', 'gold', 'lightgreen', 'purple', 'orange','silver','pink','green','red','grey','Turquoise','lightgrey']
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pos_counts, labels = pos_counts.index, autopct='%.1f%%',colors=colors)
ax1.set_title("Positive Sentiment")


#NEGATIVE PIE CHART by email_Company
neg_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']
neg_counts = neg_df['email_company'].value_counts()
neg_sorted_counts = neg_counts.sort_values(ascending=False)  
neg_others = neg_sorted_counts[neg_sorted_counts < len(neg_df) * 0.04].sum()
neg_counts = neg_counts[neg_counts > len(neg_df) * 0.04]
neg_counts['Others'] = neg_others
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.pie(neg_counts, labels = neg_counts.index, autopct='%.1f%%',colors=colors)
ax2.set_title("Negative Sentiment")
plt.show()


# Count sentiments by email_verified
pos_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'].groupby('email_verified')['textID'].count()
neg_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative'].groupby('email_verified')['textID'].count()
total_by_platform = trainSetUpdated.groupby('email_verified')['textID'].count()

# Calculate percentages
pos_percent = pos_count / total_by_platform * 100   
neg_percent = neg_count / total_by_platform * 100

#RELATIVE STACKED BAR 
sentiment_percent = pd.concat([pos_percent, neg_percent], axis=1)
sentiment_percent.columns = ['Positive', 'Negative'] 

ax = sentiment_percent.plot.bar(stacked=True,figsize=(9,6)) 
ax.set_ylabel("Percentage") 
plt.yticks(np.arange(0, 110, 10)) 
plt.yticks(np.arange(0, 110, 10), ['%d%%'% x for x in np.arange(0, 110, 10)])
plt.title("Sentiment Percentage per Email_Verified", size=16)  
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.show()


#POSITIVE PIE CHART BY EMAIL_VERIFIED
pos_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'] 
pos_counts = pos_df['email_verified'].value_counts()
colors = ['lightblue', 'gold', 'lightgreen', 'purple', 'orange','silver','pink','green','red','grey','Turquoise','lightgrey']
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pos_counts, labels = pos_counts.index, autopct='%.1f%%',colors=colors)
ax1.set_title("Positive Sentiment by email_verified")


#NEGATIVE PIE CHART BY EMAIL_VERIFIED
neg_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']
neg_counts = neg_df['email_verified'].value_counts()
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.pie(neg_counts, labels = neg_counts.index, autopct='%.1f%%',colors=colors)
ax2.set_title("Negative Sentiment by email_verified")
plt.show()



#count values
counts = trainSetUpdated['email_verified'].value_counts()

#making the graphes
plt.pie(counts, 
        labels=counts.index,  
        autopct='%1.1f%%',
        colors=['lightblue','pink','yellow']
       )
plt.title('email_verified Percentage')
plt.axis('equal') 
plt.tight_layout()
plt.legend()
plt.show() 




# Count sentiments by gender
pos_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'].groupby('gender')['textID'].count()
neg_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative'].groupby('gender')['textID'].count()
total_by_platform = trainSetUpdated.groupby('gender')['textID'].count()

# Calculate percentages
pos_percent = pos_count / total_by_platform * 100   
neg_percent = neg_count / total_by_platform * 100

#RELATIVE STACKED BAR
sentiment_percent = pd.concat([pos_percent, neg_percent], axis=1)
sentiment_percent.columns = ['Positive', 'Negative'] 

ax = sentiment_percent.plot.bar(stacked=True,figsize=(9,6)) 
ax.set_ylabel("Percentage") 
plt.yticks(np.arange(0, 110, 10)) 
plt.yticks(np.arange(0, 110, 10), ['%d%%'% x for x in np.arange(0, 110, 10)])
plt.title("Sentiment Percentage per gender", size=16)  
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.show()


#POSITIVE PIE CHART BY gender
pos_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'] 
pos_counts = pos_df['gender'].value_counts()
colors = ['lightblue', 'gold', 'lightgreen', 'purple', 'orange','silver','pink','green','red','grey','Turquoise','lightgrey']
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pos_counts, labels = pos_counts.index, autopct='%.1f%%',colors=colors)
ax1.set_title("Positive Sentiment by gender")


#NEGATIVE PIE CHART BY gender
neg_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']
neg_counts = neg_df['gender'].value_counts()
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.pie(neg_counts, labels = neg_counts.index, autopct='%.1f%%',colors=colors)
ax2.set_title("Negative Sentiment by gender")
plt.show()

#count values
counts = trainSetUpdated['gender'].value_counts()

#making the graphes
plt.pie(counts, 
        labels=counts.index,  
        autopct='%1.1f%%',
        colors=['lightblue','pink','yellow']
       )
plt.title('Gender Percentage')
plt.axis('equal') 
plt.tight_layout()
plt.legend()
plt.show() 




# Count sentiments by blue_tick
pos_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'].groupby('blue_tick')['textID'].count()
neg_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative'].groupby('blue_tick')['textID'].count()
total_by_platform = trainSetUpdated.groupby('blue_tick')['textID'].count()

# Calculate percentages
pos_percent = pos_count / total_by_platform * 100   
neg_percent = neg_count / total_by_platform * 100

#RELATIVE STACKED BAR
sentiment_percent = pd.concat([pos_percent, neg_percent], axis=1)
sentiment_percent.columns = ['Positive', 'Negative'] 

ax = sentiment_percent.plot.bar(stacked=True,figsize=(9,6)) 
ax.set_ylabel("Percentage") 
plt.yticks(np.arange(0, 110, 10)) 
plt.yticks(np.arange(0, 110, 10), ['%d%%'% x for x in np.arange(0, 110, 10)])
plt.title("Sentiment Percentage per blue_tick", size=16)  
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.show()


#POSITIVE PIE CHART BY BLUE_TICK
pos_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'] 
pos_counts = pos_df['blue_tick'].value_counts()
colors = ['lightblue', 'gold', 'lightgreen', 'purple', 'orange','silver','pink','green','red','grey','Turquoise','lightgrey']
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pos_counts, labels = pos_counts.index, autopct='%.1f%%',colors=colors)
ax1.set_title("Positive Sentiment by blue_tick")


#NEGATIVE PIE CHART BY BLUE_TICK
neg_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']
neg_counts = neg_df['blue_tick'].value_counts()
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.pie(neg_counts, labels = neg_counts.index, autopct='%.1f%%',colors=colors)
ax2.set_title("Negative Sentiment by blue_tick")
plt.show()



#count values
counts = trainSetUpdated['blue_tick'].value_counts()

#making the graphes
plt.pie(counts, 
        labels=counts.index,  
        autopct='%1.1f%%',
        colors=['lightblue','pink','yellow']
       )
plt.title('blue_tick Percentage')
plt.axis('equal') 
plt.tight_layout()
plt.legend()
plt.show() 



# Count sentiments by embedded_content
pos_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'].groupby('embedded_content')['textID'].count()
neg_count = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative'].groupby('embedded_content')['textID'].count()
total_by_platform = trainSetUpdated.groupby('embedded_content')['textID'].count()

# Calculate percentages
pos_percent = pos_count / total_by_platform * 100   
neg_percent = neg_count / total_by_platform * 100

#RELATIVE STACKED BAR
sentiment_percent = pd.concat([pos_percent, neg_percent], axis=1)
sentiment_percent.columns = ['Positive', 'Negative'] 
ax = sentiment_percent.plot.bar(stacked=True,figsize=(9,6)) 
ax.set_ylabel("Percentage") 
plt.yticks(np.arange(0, 110, 10)) 
plt.yticks(np.arange(0, 110, 10), ['%d%%'% x for x in np.arange(0, 110, 10)])
plt.title("Sentiment Percentage per embedded_content", size=16)  
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.show()


#POSITIVE PIE CHART BY EMBEDDED_CONTENT
pos_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'positive'] 
pos_counts = pos_df['embedded_content'].value_counts()
colors = ['lightblue', 'gold', 'lightgreen', 'purple', 'orange','silver','pink','green','red','grey','Turquoise','lightgrey']
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pos_counts, labels = pos_counts.index, autopct='%.1f%%',colors=colors)
ax1.set_title("Positive Sentiment by embedded_content")


#NEGATIVE PIE CHART BY EMBEDDED_CONTENT
neg_df = trainSetUpdated[trainSetUpdated['sentiment'] == 'negative']
neg_counts = neg_df['embedded_content'].value_counts()
neg_counts['Others'] = neg_others
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.pie(neg_counts, labels = neg_counts.index, autopct='%.1f%%',colors=colors)
ax2.set_title("Negative Sentiment by embedded_content")
plt.show()

#count values
counts = trainSetUpdated['embedded_content'].value_counts()

#making the graphes
plt.pie(counts, 
        labels=counts.index,  
        autopct='%1.1f%%',
        colors=['lightblue','pink','yellow','orange','lightgreen']
       )
plt.title('embedded_content Percentage')
plt.axis('equal') 
plt.tight_layout()
plt.legend()
plt.show() 




#CORR MATRIX- Intervariable dependencies
correlation_matrix = trainSetUpdated[['char_count', 'word_count']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix between char_count and word_count")
plt.show()




