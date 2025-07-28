import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data=[['ham',"Hey, are we still on for dinner tonight?"],['spam',"WINNER! Free cruise. Call now!"],
      ['ham',"I'll call you back in 10 minutes."],['spam',"URGENT! Account suspended. Verify now."],
      ['ham',"Meeting at 3 PM."],['spam',"Claim your free ringtone, text WIN!"]]
df=pd.DataFrame(data,columns=['label','text']);df['label']=df['label'].map({'ham':0,'spam':1})

# Train/Test split
X_train,X_test,y_train,y_test=train_test_split(df['text'],df['label'],test_size=0.3,random_state=42)

# Train model
vec=CountVectorizer(stop_words='english');X_train_vec=vec.fit_transform(X_train);X_test_vec=vec.transform(X_test)
model=MultinomialNB().fit(X_train_vec,y_train)

# Predictions
print("Accuracy:",model.score(X_test_vec,y_test))
msgs=["URGENT! Account suspended.","Hi, how are you?"]
for m,p in zip(msgs,model.predict(vec.transform(msgs))):
    print(f"'{m}' =>",["ham","spam"][p])
