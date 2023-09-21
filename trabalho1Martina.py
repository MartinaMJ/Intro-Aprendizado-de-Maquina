# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:55:54 2022

@author: marti
"""
# Modelo preditivo para aprovação de crédito # 

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# Definindo os dados #

dados = pd.read_csv('conjunto_de_treinamento.csv')
dadosTeste = pd.read_csv('conjunto_de_teste.csv')

# Descobrindo as variáveis categóricas, ajuda a escolher as variáveis úteis #

categoricas = [ x for x in dados.columns if dados[x].dtype == 'object' or x == 'codigo_area_telefone_residencial'
               or x == 'codigo_area_telefone_trabalho']

print (categoricas)

# Descobrindo a cardinalidade #

for variavel in categoricas:
    print ('\n%15s: '%variavel , "%4d categorias" % len(dados[variavel].unique()))
    print (dados[variavel].unique(), '\n')
    
# Serão excluídas as de cardinalidade muito altas e/ou com resultados incoerentes #

dados = dados.drop (['sexo', 'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho', 'possui_telefone_celular',
                     'id_solicitante', 'estado_onde_nasceu', 'estado_onde_reside', 'estado_onde_trabalha', 'qtde_contas_bancarias_especiais'], axis = 1)

dadosTeste = dadosTeste.drop (['sexo', 'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho', 'possui_telefone_celular',
                    'id_solicitante', 'estado_onde_nasceu', 'estado_onde_reside', 'estado_onde_trabalha', 'qtde_contas_bancarias_especiais'], axis = 1)

# Binarizando 
binarizador = LabelBinarizer()

for contador in ['possui_telefone_residencial', 'vinculo_formal_com_empresa', 'possui_telefone_trabalho']:
    dados [contador] = binarizador.fit_transform(dados[contador])
    dadosTeste [contador] = binarizador.fit_transform(dadosTeste[contador])
    
# One Hot Encoding para variaveis com mais de 3 categorias #

dados = pd.get_dummies(dados, columns = ['forma_envio_solicitacao'])
dadosTeste = pd.get_dummies(dadosTeste, columns = ['forma_envio_solicitacao'])

selecionados = ['produto_solicitado', 'dia_vencimento', 'tipo_endereco', 'idade',
       'estado_civil', 'qtde_dependentes', 'grau_instrucao', 'nacionalidade',
       'possui_telefone_residencial', 'tipo_residencia', 'meses_na_residencia',
       'possui_email', 'renda_mensal_regular', 'renda_extra',
       'possui_cartao_visa', 'possui_cartao_mastercard',
       'possui_cartao_diners', 'possui_cartao_amex', 'possui_outros_cartoes',
       'qtde_contas_bancarias',
       'valor_patrimonio_pessoal', 'possui_carro',
       'vinculo_formal_com_empresa', 'possui_telefone_trabalho',
       'meses_no_trabalho', 'profissao', 'ocupacao', 'profissao_companheiro',
       'grau_instrucao_companheiro', 'local_onde_reside',
       'local_onde_trabalha','forma_envio_solicitacao_correio', 'forma_envio_solicitacao_internet',
       'forma_envio_solicitacao_presencial'] 


alvo = 'inadimplente'

# Os espaços vazios e NotANumber serão substituídos por zero

dados = dados.fillna(0)
embaralhado = dados.sample(frac=1, random_state = 12345)
dadosTeste = dadosTeste.fillna(0)

matrizX = embaralhado.loc[:,embaralhado.columns!='inadimplente'].values
matrizY = embaralhado.loc[:,embaralhado.columns=='inadimplente'].values
matrizXTeste = dadosTeste.loc[:,dadosTeste.columns!='inadimplente'].values

scaler = MinMaxScaler()

linhasTreino = 20000

x_treino = matrizX[:linhasTreino,:]
y_treino = matrizY[:linhasTreino].ravel()
x_teste = matrizXTeste

x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.fit_transform(x_teste)

classificador = KNeighborsClassifier(n_neighbors=40)

classificador = classificador.fit(x_treino, y_treino)

print ('resposta Treino:')

y_resposta_treino = classificador.predict(x_treino)

print (y_resposta_treino)

print ('resposta Teste:')

y_resposta_teste = classificador.predict(x_teste)

print (y_resposta_teste)

print ('AVALIAÇÃO DE RESULTADOS POR VALIDAÇÃO CRUZADA')
kfold  = KFold(n_splits=5, shuffle=False)
resultado = cross_val_score(classificador, x_treino, y_treino, cv = kfold, scoring = 'accuracy')
print(" Acurácia para cada bloco K-Fold : {0}".format(resultado))
print("Média das acurácias para Cross-Validation K-Fold: {0}".format(resultado.mean()))

# Salvando em formato CSV
#arqvTeste = pd.read_csv('conjunto_de_teste.csv')

#resposta_KNN_Martina = pd.DataFrame({'id_solicitante':arqvTeste.pop('id_solicitante'), 'inadimplente':np.squeeze(np.int16(y_resposta_teste))})

#resposta_KNN_Martina.to_csv("resposta_KNN_Martina.csv",index=False)
