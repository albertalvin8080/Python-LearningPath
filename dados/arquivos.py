import pandas as pd

d = {"Inquisidores":['Prometheus','Luqinhas'],
     'Livros':['como cair na vida','nao se mova agora']}
data = pd.DataFrame(d)

data.to_csv('C:/Users/Albert/Documents/A_Programacao/Python/dados/arq.csv',
            index=False)

print(pd.read_csv('C:/Users/Albert/Documents/A_Programacao/Python/dados/arq.csv'))

