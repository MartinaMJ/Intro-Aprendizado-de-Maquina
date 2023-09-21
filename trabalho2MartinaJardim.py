# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 10:07:44 2022

@author: marti
"""

# Será utilizado o regressor KNN

from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import math

dadosTreino = pd.read_csv('conjunto_de_treinamento.csv')
dadosTeste = pd.read_csv('conjunto_de_teste.csv')

# Analisando as variaveis #

categoricas = [ x for x in dadosTreino.columns if dadosTreino[x].dtype == 'object']
print (categoricas)

for variavel in categoricas:
    print ('\n%15s: '%variavel , "%4d categorias" % len(dadosTreino[variavel].unique()))
    print (dadosTreino[variavel].unique(), '\n')

# Binarizando 
binarizador = LabelBinarizer()

for contador in ['tipo_vendedor']:
    dadosTreino [contador] = binarizador.fit_transform(dadosTreino[contador])
    dadosTeste [contador] = binarizador.fit_transform(dadosTeste[contador])
    
# One Hot Encoding #
dadosTreino = pd.get_dummies(dadosTreino, columns = ['tipo'])
dadosTeste = pd.get_dummies(dadosTeste, columns = ['tipo'])

# Limitando os preços entre 50 mil e 5 milhões

dadosTreino = dadosTreino[(dadosTreino['preco']>= 50000) & (dadosTreino['preco']<= 5000000)]

# Classificando os bairros com base em seu IDH, com base em dados de 2005
# IDH Muito Baixo = 0.666
# IDH Baixo = 0.7125
# IDH Médio = 0.7455
# IDH Alto = 0.815
# IDH Muito Alto = 0.9145

bairros = {'Imbiribeira' : 0.815, 'Casa Amarela' : 0.9145, 'Encruzilhada' : 0.9145,
           'Boa Viagem' : 0.666, 'Rosarinho' : 0.9145, 'Boa Vista' : 0.9145,
           'Espinheiro' : 0.9145, 'Tamarineira' : 0.9145, 'Gracas' : 0.9145,
           'Madalena' : 0.9145, 'Parnamirim' : 0.9145, 'S Jose' : 0.7125,
           'Setubal' : 0.9145, 'Arruda' : 0.9145, 'Pina' : 0.7125, 'Beira Rio' : 0.7125,
           'Caxanga' : 0.7455, 'Casa Forte' : 0.7455, 'Prado' : 0.7455, 
           'Iputinga' : 0.7125, 'Campo Grande' : 0.7455, 'Dois Irmaos': 0.666, 
           'Torreao' : 0.9145, 'Ilha do Retiro' : 0.7455, 'Areias' : 0.815, 
           'Varzea' : 0.7455, 'Cordeiro' : 0.9145, 'Santana': 0.9145, 'Torre' : 0.9145,
           'Barro' : 0.7125, 'Poco da Panela' : 0.9145, 'Ipsep' : 0.9145, 
           'Apicupos' : 0.666, 'Aflitos' : 0.9145, 'Poco' : 0.9145, 'Apipucos' : 0.666,
           'Derby' : 0.9145, 'Cid Universitaria' : 0.7455, 'Bongi' : 0.7455, 'Jaqueira' : 0.9145, 
           'Sto Amaro' : 0.7455, 'Tejipio' : 0.815, 'Recife' : 0.7125, 
           'Monteiro' : 0.815, 'Macaxeira' : 0.666, 'Sancho' : 0.7455, 
           'Afogados' : 0.7455, 'San Martin' : 0.815, 'Cajueiro' : 0.815,
           'Hipodromo' : 0.9145, 'Guabiraba' : 0.666, 'Engenho do Meio' : 0.815,
           'Piedade' : 0.7455 , 'Jd S Paulo' : 0.815, 'Lagoa do Araca' : 0.815 , 
           'Ponto de Parada' : 0.9145, 'Ilha do Leite' : 0.9145, 'Estancia' : 0.815,
           'Paissandu' : 0.9145, 'Zumbi' : 0.9145, 'Agua Fria' : 0.7455, 
           'Benfica' : 0.9145, 'Soledade' : 0.815, 'Centro' : 0.7455 , 'Sto Antonio' : 0.7125,
           'Coelhos' : 0.9145, 'Cohab' : 0.7125, 'Ibura' : 0.7125, 'Beberibe' : 0.7125, 'Fundao' : 0.666 }

dadosTreino = dadosTreino.replace(bairros)
dadosTeste = dadosTeste.replace(bairros)

# Classificando os diferenciais

diferenciais = {'campo de futebol e copa':'futebol e outros',
 'campo de futebol e esquina':'futebol e outros',
 'campo de futebol e estacionamento visitantes':'futebol e outros',
 'campo de futebol e playground':'futebol e outros',
 'campo de futebol e quadra poliesportiva':'futebol e outros',
 'campo de futebol e salao de festas':'futebol e outros',
 'campo de futebol e sala de ginastica':'futebol e outros',
 'children care':'apenas children care',
 'children care e playground':'children care e outros',
 'churrasqueira':'apenas churrasco',
 'churrasqueira e campo de futebol':'churrasco e outros',
 'churrasqueira e copa':'churrasco e outros',
 'churrasqueira e esquina':'churrasco e outros',
 'churrasqueira e estacionamento visitantes':'churrasco e outros',
 'churrasqueira e frente para o mar':'churrasco e outros',
 'churrasqueira e playground':'churrasco e outros',
 'churrasqueira e sala de ginastica':'churrasco e outros',
 'churrasqueira e salao de festas':'churrasco e outros',
 'churrasqueira e sauna':'churrasco e outros',
 'churrasqueira e children care':'churrasco e outros',
 'copa':'apenas copa',
 'copa e esquina':'copa e outros',
 'copa e estacionamento visitantes':'copa e outros',
 'copa e playground':'copa e outros',
 'copa e quadra poliesportiva':'copa e outros',
 'copa e sala de ginastica':'copa e outros',
 'copa e salao de festas':'copa e outros',
 'copa e hidromassagem':'copa e outros',
 'esquina':'apenas esquina',
 'esquina e estacionamento visitantes':'esquina e outros',
 'esquina e playground':'esquina e outros',
 'esquina e quadra poliesportiva':'esquina e outros',
 'esquina e sala de ginastica':'esquina e outros',
 'esquina e salao de festas':'esquina e outros',
 'estacionamento visitantes':'apenas estacionamento visitantes',
 'estacionamento visitantes e playground':'estacionamento visitantes e outros',
 'estacionamento visitantes e sala de ginastica':'estacionamento visitantes e outros',
 'estacionamento visitantes e salao de festas':'estacionamento visitantes e outros',
 'estacionamento visitantes e hidromassagem':'estacionamento visitantes e outros',
 'estacionamento visitantes e salao de jogos':'estacionamento visitantes e outros',
 'frente para o mar':'apenas frente para o mar',
 'frente para o mar e campo de futebol':'frente para o mar e outros',
 'frente para o mar e copa':'frente para o mar e outros',
 'frente para o mar e esquina':'frente para o mar e outros',
 'frente para o mar e playground':'frente para o mar e outros',
 'frente para o mar e quadra poliesportiva':'frente para o mar e outros',
 'frente para o mar e salao de festas':'frente para o mar e outros',
 'frente para o mar e children care':'frente para o mar e outros',
 'frente para o mar e hidromassagem':'frente para o mar e outros',
 'nenhum':'nenhum',
 'piscina':'apenas piscina',
 'piscina e campo de futebol':'piscina e outros',
 'piscina e children care':'piscina e outros',
 'piscina e churrasqueira':'piscina e outros',
 'piscina e copa':'piscina e outros',
 'piscina e esquina':'piscina e outros',
 'piscina e estacionamento visitantes':'piscina e outros',
 'piscina e frente para o mar':'piscina e outros',
 'piscina e hidromassagem':'piscina e outros',
 'piscina e playground':'piscina e outros',
 'piscina e quadra de squash':'piscina e outros',
 'piscina e quadra poliesportiva':'piscina e outros',
 'piscina e sala de ginastica':'piscina e outros',
 'piscina e salao de festas':'piscina e outros',
 'piscina e salao de jogos':'piscina e outros',
 'piscina e sauna':'piscina e outros',
 'playground':'apenas playground',
 'playground e quadra poliesportiva':'playground e outros',
 'playground e sala de ginastica':'playground e outros',
 'playground e salao de festas':'playground e outros',
 'playground e salao de jogos':'playground e outros',
 'quadra poliesportiva':'apenas quadra poliesportiva',
 'quadra poliesportiva e salao de festas':'quadra e outros',
 'sala de ginastica':'apenas sala de ginastica',
 'sala de ginastica e salao de festas':'ginastica e outros',
 'sala de ginastica e salao de jogos':'ginastica e outros',
 'salao de festas':'apenas salao de festas',
 'salao de festas e salao de jogos':'festa e outros',
 'salao de festas e vestiario':'festa e outros',
 'salao de jogos':'apenas salao de jogos',
 'sauna':'apenas sauna',
 'sauna e campo de futebol':'sauna e outros',
 'sauna e copa':'sauna e outros',
 'sauna e esquina':'sauna e outros',
 'sauna e frente para o mar':'sauna e outros',
 'sauna e playground':'sauna e outros',
 'sauna e quadra poliesportiva':'sauna e outros',
 'sauna e sala de ginastica':'sauna e outros',
 'sauna e salao de festas':'sauna e outros',
 'vestiario':'apenas vestiario',
 'futebol e sala de ginastica':'futebol e outros',
 'mar e hidromassagem':'frente para o mar e outros',
 'hidromassagem e salao de festas':'festa e outros'}

dadosTreino = dadosTreino.replace(diferenciais)
dadosTeste = dadosTeste.replace(diferenciais)

#One hot encoding
dadosTreino = pd.get_dummies(dadosTreino,columns = ['diferenciais'])
dadosTeste = pd.get_dummies(dadosTeste,columns = ['diferenciais'])

#Dropando repetições
dadosTreino = dadosTreino.drop (['Id', 'churrasqueira', 'piscina', 'playground', 'sauna',
                                 'quadra', 's_festas', 's_jogos', 's_ginastica', 'vista_mar'], axis = 1)
dadosTeste = dadosTeste.drop (['Id', 'churrasqueira', 'piscina', 'playground', 'sauna',
                                 'quadra', 's_festas', 's_jogos', 's_ginastica', 'vista_mar'], axis = 1)

dadosTreino = dadosTreino.replace(diferenciais)
dadosTeste = dadosTeste.replace(diferenciais)

colunasNovas = dadosTreino.columns
for col in colunasNovas:
  print('%10s = %6.3f' % (col, pearsonr(dadosTreino[col],dadosTreino['preco'])[0]))
  
# Defnindo os Arrays

matrizXTreino = dadosTreino.drop(['preco','tipo_Quitinete','diferenciais_apenas children care',
                                  'area_extra', 'estacionamento', 'area_extra', 'estacionamento',
                                  'tipo_Loft','diferenciais_children care e outros',
                                  'diferenciais_apenas churrasco', 'diferenciais_churrasco e outros',
                                  'diferenciais_apenas copa', 'diferenciais_copa e outros', 'diferenciais_apenas esquina',
                                  'diferenciais_esquina e outros', 'diferenciais_apenas estacionamento visitantes',
                                  'diferenciais_estacionamento visitantes e outros', 'diferenciais_festa e outros',
                                  'diferenciais_apenas frente para o mar', 'diferenciais_futebol e outros',
                                  'diferenciais_ginastica e outros', 'diferenciais_apenas piscina',
                                  'diferenciais_piscina e outros', 'diferenciais_apenas playground',
                                  'diferenciais_playground e outros', 'diferenciais_apenas quadra poliesportiva',
                                  'diferenciais_quadra e outros', 'diferenciais_apenas sala de ginastica',
                                  'diferenciais_apenas salao de festas', 'diferenciais_apenas salao de jogos',
                                  'diferenciais_apenas sauna', 'diferenciais_sauna e outros', 'diferenciais_apenas vestiario'], axis=1)
matrizYTreino = dadosTreino['preco']
matrizXTeste = dadosTeste.drop(['area_extra', 'estacionamento', 'area_extra',
                                  'tipo_Loft', 'diferenciais_apenas churrasco', 'diferenciais_churrasco e outros',
                                  'diferenciais_apenas copa', 'diferenciais_copa e outros', 'diferenciais_apenas esquina',
                                  'diferenciais_esquina e outros', 'diferenciais_apenas estacionamento visitantes',
                                  'diferenciais_estacionamento visitantes e outros', 'diferenciais_festa e outros',
                                  'diferenciais_apenas frente para o mar', 'diferenciais_futebol e outros',
                                  'diferenciais_ginastica e outros', 'diferenciais_apenas piscina',
                                  'diferenciais_piscina e outros', 'diferenciais_apenas playground',
                                  'diferenciais_playground e outros', 'diferenciais_quadra e outros', 'diferenciais_apenas sala de ginastica',
                                  'diferenciais_apenas salao de festas', 'diferenciais_apenas salao de jogos',
                                  'diferenciais_apenas sauna', 'diferenciais_sauna e outros', 'diferenciais_apenas vestiario'], axis=1)

#Escalando
scaler = StandardScaler()
scaler.fit(matrizXTreino)

matrizXTreino = scaler.transform(matrizXTreino)
matrizXTeste = scaler.transform(matrizXTeste)

#Regressor KNN
regressorKNN = KNeighborsRegressor (n_neighbors = 5)
regressorKNN = regressorKNN.fit(matrizXTreino, matrizYTreino)
    
print ('AVALIAÇÃO DE RESULTADOS POR VALIDAÇÃO CRUZADA')

kfold  = KFold(n_splits=5, shuffle=True)
resultado = cross_val_score(regressorKNN, matrizXTreino, matrizYTreino, cv = kfold)
print("K-Fold (R^2) Scores: {0}".format(resultado))
print("Média dos R^2 para Cross-Validation K-Fold: {0}".format(resultado.mean()))

y_resposta_treino = regressorKNN.predict(matrizXTreino)
mse_treino = mean_squared_error(matrizYTreino, y_resposta_treino)
rmse_treino = math.sqrt(mse_treino)
r2_treino = r2_score(matrizYTreino, y_resposta_treino)
rmspe_treino = (np.sqrt(np.mean(np.square((matrizYTreino - y_resposta_treino) / matrizYTreino))))
print (f' MSE Treino: {mse_treino}, RMSE Treino: {rmse_treino}, R2 Treino: {r2_treino}, RMSPE Treino: {rmspe_treino}')
    
y_resposta_teste = regressorKNN.predict(matrizXTeste)
print (f'Resposta Teste : {y_resposta_teste}')
Id = pd.read_csv('conjunto_de_teste.csv')
respostaKNNMartina2 = pd.DataFrame({'Id':Id.pop('Id'), 'preco':np.squeeze((y_resposta_teste))})
respostaKNNMartina2.to_csv("respostaKNNMartina2.csv", index=False)

