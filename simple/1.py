import pandas as pd
def find_s(examples):
    step=1;
    hypo=['0']*len(examples[0][0])
    print(f"Initial Hypothesis is {hypo}\n")
    for instance, label in examples:
        if label=='Yes':
             if hypo==['0']*len(hypo):
                 hypo=instance.copy()
                 print(f"Step {step} hypothesis is {hypo}\n")
                 
             else:
                 for i in range(len(hypo)):
                     if hypo[i]!=instance[i]:
                         hypo[i]='?'
                 print(f"Step {step} hypothesis is {hypo}\n")        
             step+=1                
    else:
         print("skipping negative example\n")

df=pd.read_csv("training_data.csv")
data=[]
for index, row in df.iterrows():
    features=row.iloc[:-1].tolist()
    label=row.iloc[-1]
    data.append((features,label))


find_s(data)
    
        
   
    
            
       
    
            
      
