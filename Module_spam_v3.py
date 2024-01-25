import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, OrdinalEncoder, StandardScaler
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import ParameterGrid



# Classe definissant les chemins du labyrinthe
class model_IA :

    # Constructeur, avec en argument le point d'apparition de la tuille
    def __init__(self, df_init, model_parametre_init, test=True):
        
        self.df= df_init
        self.model_parametre = model_parametre_init
        self.pipeline_model = make_pipeline ( self.model_parametre ) 

        self.df_papel = self.taitement_na_duplic(self.df)
        self.y_papel = self.df_papel['classification']
        self.x_papel = self.df_papel['sms']

        self.df = self.preprocessing_df(self.df)
        self.x_train, self.y_train, self.x_test, self.y_test = self.train_et_test(self.df)
        self.model_fit (self.pipeline_model, self.x_train, self.y_train)
        self.model_predict(self.pipeline_model, self.x_train, self.y_train, self.x_test, self.y_test)

    
    ####
    # fonction :
    ####
    def test_new_csv (self, df_new) :
        """
        -------------------------
        """
        self.df_new = self.preprocessing_df(df_new)
        self.x_new_test_test, self.y_new_test_test = self.train_et_test_test(self.df_new)
        self.model_predict(self.pipeline_model, self.x_train, self.y_train, self.x_new_test_test, self.y_new_test_test)

        return self.x_new_test_test, self.y_new_test_test, self.y_pred, self.score, self.confusion_matrix, self.N, self.train_score, self.val_score 

    def taitement_na_duplic (self, df) :
        """
        entrée : un data frame
        sortie : 2 data frame = 'principal' et 'na'
        ---------------------------
        """
        df = df.drop_duplicates()
        df = df.dropna()
        df.rename(columns={0:'classification', '0':'classification'}, inplace=True)
        df.rename(columns={1:'sms', '1':'sms'}, inplace=True)
        return df
    
    def mot_cle_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        mot_cles = ['URGENT!', 'Quiz!', 'YOU!', 'Txt:', 'now!', 'Call ', 'Win', 'WINNER', '!!', 'For sale', 'FREE!', 'PRIVATE!', 'Account', 'Latest News!']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def argent_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        mot_cles = ['£', '€', '\$']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def telephone_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        crée le pattern des numero de tel
        recherche dans une chane de carractère si je trouve le patern    
        """
        pattern = re.compile(r"(\+\d{1,3})?\s?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,4}")
        match = re.search(pattern, sms)
        return bool(match)

    def email_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        je crée le pattern des e-mail
        je recherche dans la colonne 'sms' si je trouve le patern    
        """
        pattern = r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
        match = re.findall(pattern, sms)
        return bool(match)

    def lien_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        mot_cles = ['http', 'https', 'www.', 'click here']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def mot_maj_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : boolean
        ---------------------
        je crée le pattern des majuscules
        je recherche dans la colonne 'sms' si je trouve le patern  
        """
        pattern = "[A-Z]{3}"
        match = re.findall(pattern, sms)
        return bool(match)

    def long_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : int
        ---------------------
        je mesure la taille de chaque ligne de la colonne 'sms'
        """
        return int(len(sms))

    def nb_mot_posible (self, sms) :
        """
        entrée : chaine de carractaire
        sortie : int
        ---------------------
        je mesure le nombre de mots de chaque ligne de la colonne 'sms'
        """
        list_of_words = sms.split()
        return int(len(list_of_words))

    def vectorisation_df (self, df) :
        """
        entrée : un data frame
        sortie : un data frame
        ---------------------------
        je crée la colonne 'mot_cles' grace à la fonction 'mot_cle_posible'
        je crée la colonne 'argent' grace à la fonction 'argent_posible'
        je crée la colonne 'telephone' grace à la fonction 'telephone_posible'
        je crée la colonne 'email' grace à la fonction 'email_posible'
        je crée la colonne 'lien' grace à la fonction 'lien_posible'
        je crée la colonne 'maj' grace à la fonction 'mot_maj_posible'
        je crée la colonne 'long' grace à la fonction 'long_posible'
        """    
        df['mot_cles'] = df['sms'].apply(self.mot_cle_posible)
        df['argent'] = df['sms'].apply(self.argent_posible)
        df['telephone'] = df['sms'].apply(self.telephone_posible)
        df['email'] = df['sms'].apply(self.email_posible)
        df['lien'] = df['sms'].apply(self.lien_posible)
        df['maj'] = df['sms'].apply(self.mot_maj_posible)
        df['long'] = df['sms'].apply(self.long_posible)
        df['mot'] = df['sms'].apply(self.nb_mot_posible)
        return df

    def preprocessing_df (self, df):
        """
        entrée : un data frame
        sortie : 4 data frame et les fitures et tagets
        ---------------------------
        je lance le netoyage des données grace à la fonction taitement_na_duplic
        je lance l'encodage des données grace à la fonction encodage_df
        je lance la création des fiture et targets grace à la fonction traine_et_test
        """
        df = self.taitement_na_duplic(df)
        df = self.vectorisation_df (df)
        label_encod = LabelEncoder()
        robust_scal = RobustScaler()
        df['classification'] = label_encod.fit_transform(df['classification'])
        df['mot_cles'] = label_encod.fit_transform(df['mot_cles'])
        df['argent'] = label_encod.fit_transform(df['argent'])
        df['telephone'] = label_encod.fit_transform(df['telephone'])
        df['email'] = label_encod.fit_transform(df['email'])
        df['maj'] = label_encod.fit_transform(df['maj'])
        df['lien'] = label_encod.fit_transform(df['lien'])
        robust_scal = RobustScaler()
        standar_scal = StandardScaler()
        norm_long = df[['long' , 'mot']]
        norm_long = standar_scal.fit_transform(norm_long)
        df[['long' , 'mot']] = norm_long
        return df

    def train_et_test (self, df) :
        """
        entrée : un data frame, pourcentage de valeur à metre dans df_test
        sortie : 2 data frame = 'traine' et 'test' et leur x et y respectif
        ---------------------------
        avec train_test_split, je crée sépare en 2 le df : trainSet et  testSet
        je crée le y_train
        je crée le y_test
        je crée le x_train
        je crée le x_test
        """
        trainSet, testSet = train_test_split(df, test_size=0.2, random_state=0, stratify=df['classification'])
        trainSet.drop('sms', axis=1, inplace=True)
        testSet.drop('sms', axis=1, inplace=True)
        y_train = trainSet['classification']
        y_test = testSet['classification']
        x_train = trainSet.drop(['classification'], axis=1)
        x_test = testSet.drop(['classification'], axis=1)
        return x_train, y_train, x_test, y_test

    def train_et_test_test (self, df) :
        """
        entrée : un data frame, pourcentage de valeur à metre dans df_test
        sortie : 2 data frame = 'traine' et 'test' et leur x et y respectif
        ---------------------------
        avec train_test_split, je crée sépare en 2 le df : trainSet et  testSet
        je crée le y_train
        je crée le y_test
        je crée le x_train
        je crée le x_test
        """
        testSet = df.drop('sms', axis=1)
        y_test_test = testSet['classification']
        x_test_test = testSet.drop(['classification'], axis=1)
        return x_test_test, y_test_test

    def model_fit (self, papeline, x, y) :
        """
        entrée : un data frame, pourcentage de valeur à metre dans df_test
        sortie : 2 data frame = 'traine' et 'test'
        ---------------------------
        avec train_test_split, je crée sépare en 2 le df
        pourcent est le pourcentage de valeur à metre dans df_test
        """
        papeline.fit(x, y)

    def model_predict (self, papeline, x_train, y_train, x_test, y_test,) :
        """
        entrée : un data frame, pourcentage de valeur à metre dans df_test
        sortie : 2 data frame = 'traine' et 'test'
        ---------------------------
        avec train_test_split, je crée sépare en 2 le df
        pourcent est le pourcentage de valeur à metre dans df_test
        """
        self.y_pred = papeline.predict(x_test)
        self.score = accuracy_score(y_test, self.y_pred)

        self.confusion_matrix = confusion_matrix(y_test, self.y_pred)

        self.classification_report = classification_report(y_test, self.y_pred)

        self.N, self.train_score, self.val_score = learning_curve(papeline, x_train, y_train, scoring='f1',
                                                train_sizes=np.linspace(0.1, 1, 10))