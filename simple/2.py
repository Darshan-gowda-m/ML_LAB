import pandas as pd
df=pd.read_csv("training_data.csv")
X,Y=df.iloc[:,:-1].values,df.iloc[:,-1].values

S=['0']*X.shape[1]; G=[['?']*X.shape[1]]
for step,(x,y) in enumerate(zip(X,Y),1):
    if y=='Yes':
        for i in range(len(S)):
            if S[i]=='0': S[i]=x[i]
            elif S[i]!=x[i]: S[i]='?'
        G=[g for g in G if all(g[i] in ['?',x[i]] for i in range(len(x)))]
    else:
        newG=[]
        for g in G:
            for i in range(len(S)):
                if g[i]=='?' and S[i]!=x[i] and S[i]!='0':
                    ng=g.copy(); ng[i]=S[i]; newG.append(ng)
        G=newG or G
    print(f"Step {step}:")
    print("  Specific:",S)
    print("  General :",G,"\n")
