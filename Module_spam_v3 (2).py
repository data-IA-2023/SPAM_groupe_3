import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
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
        
        # enregistre les parametres d'entrée comme valeur de classe
        self.df= df_init
        self.model_parametre = model_parametre_init

        # prépare les données x et y pour le train_test_split
        self.df_papel = self.taitement_na_duplic(self.df)
        y_papel = self.df_papel['classification']
        x_papel = self.df_papel['sms']

        x_papel_df = x_papel.to_frame()
        x_papel_vect = self.vectorisation_df(x_papel_df)

        # x et y utilisé pour le train_test_split
        self.x_papel_vect_2 = x_papel_vect.drop('sms', axis=1)
        self.y_papel_encoder = LabelEncoder().fit_transform(y_papel)

        # train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split ( self.x_papel_vect_2, self.y_papel_encoder, train_size = 0.80, test_size = 0.20, random_state = 123 )
        
        # Catégorise les colonnes de x pour le make_column_transformer
        norm_num = ['long' , 'mot']
        bool_one_hot = ['mot_cles','argent','telephone','email','lien','maj']

        # pipeline de transformation
        one_hot_encoder_pip = make_pipeline ( OneHotEncoder() )
        min_max_scaler_pip = make_pipeline ( MinMaxScaler () )

        # make_column_transformer
        transform_colonne = make_column_transformer (( one_hot_encoder_pip, bool_one_hot ), 
                                                     ( min_max_scaler_pip, norm_num ))

        # pipeline principale
        self.model_pip = make_pipeline(transform_colonne, self.model_parametre)
        
    ####
    # fonction :
    ####
        
    def entrainement_test(self) :
        
        # entrainement de la pipeline
        self.model_pip.fit( self.x_train, self.y_train )

        # résultat de la pipeline
        self.y_pred = self.model_pip.predict( self.x_test )
        self.model_pip.score( self.x_test, self.y_test )

        self.score = accuracy_score(self.y_test, self.y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)

        self.N, self.train_score, self.val_score = learning_curve(self.model_pip, self.x_train, self.y_train, scoring='f1',
                                                train_sizes=np.linspace(0.1, 1, 10))
    
    def test_new_csv (self, df_new) :
        """
        -------------------------
        """
        # prépare les données x et y pour le train_test_split
        df_new_papel = self.taitement_na_duplic(df_new)
        y_papel = self.df_new_papel['classification']
        x_papel = self.df_new_papel['sms']

        x_papel_df = x_papel.to_frame()
        x_papel_vect = self.vectorisation_df(x_papel_df)

        # x et y utilisé pour le train_test_split
        self.x_new = x_papel_vect.drop('sms', axis=1)
        self.y_new = LabelEncoder().fit_transform(y_papel)

        # résultat de la pipeline
        self.y_pred = self.model_pip.predict( self.x_new )
        self.model_pip.score( self.x_new, self.y_new )

        self.score = accuracy_score(self.y_new, self.y_pred)
        self.confusion_matrix = confusion_matrix(self.y_new, self.y_pred)

        self.N, self.train_score, self.val_score = learning_curve(self.model_pip, self.x_train, self.y_train, scoring='f1',
                                                train_sizes=np.linspace(0.1, 1, 10))

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
        entrée : chaine de caractere
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le pattern  
        """
        mot_cles = ['URGENT!', 'Quiz!', 'YOU!', 'Txt:', 'now!', 'Call ', 'Win', 'WINNER', '!!', 'For sale', 'FREE!', 'PRIVATE!', 'Account', 'Latest News!']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def argent_posible (self, sms) :
        """
        entrée : chaine de caractere
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le pattern  
        """
        mot_cles = ['£', '€', '\$']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def telephone_posible (self, sms) :
        """
        entrée : chaine de caractere
        sortie : boolean
        ---------------------
        crée le pattern des numero de tel
        recherche dans une chane de carractère si je trouve le pattern    
        """
        pattern = re.compile(r"(\+\d{1,3})?\s?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,4}")
        match = re.search(pattern, sms)
        return bool(match)

    def email_posible (self, sms) :
        """
        entrée : chaine de caractere
        sortie : boolean
        ---------------------
        je crée le pattern des e-mail
        je recherche dans la colonne 'sms' si je trouve le pattern    
        """
        pattern = r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
        match = re.findall(pattern, sms)
        return bool(match)

    def lien_posible (self, sms) :
        """
        entrée : chaine de caractere
        sortie : boolean
        ---------------------
        j'ai une liste de mots clés
        je crée le pattern des mots clés
        je recherche dans la colonne 'sms' si je trouve le pattern  
        """
        mot_cles = ['http', 'https', 'www.', 'click here']
        pattern = re.compile(r"(?=("+'|'.join(mot_cles)+r"))", re.IGNORECASE)
        match = re.findall(pattern, sms)
        return bool(match)

    def mot_maj_posible (self, sms) :
        """
        entrée : chaine de caractere
        sortie : boolean
        ---------------------
        je crée le pattern des majuscules
        je recherche dans la colonne 'sms' si je trouve le pattern  
        """
        pattern = "[A-Z]{3}"
        match = re.findall(pattern, sms)
        return bool(match)

    def long_posible (self, sms) :
        """
        entrée : chaine de carractere
        sortie : int
        ---------------------
        je mesure la taille de chaque ligne de la colonne 'sms'
        """
        return int(len(sms))

    def nb_mot_posible (self, sms) :
        """
        entrée : chaine de caractere
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
        je crée la colonne 'mot_cles' grâce à la fonction 'mot_cle_posible'
        je crée la colonne 'argent' grâce à la fonction 'argent_posible'
        je crée la colonne 'telephone' grâce à la fonction 'telephone_posible'
        je crée la colonne 'email' grâce à la fonction 'email_posible'
        je crée la colonne 'lien' grâce à la fonction 'lien_posible'
        je crée la colonne 'maj' grâce à la fonction 'mot_maj_posible'
        je crée la colonne 'long' grâce à la fonction 'long_posible'
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
    
    def detetion_de_spam(self, sms) :
    
        x = np.array([sms]).reshape(1, 1)

        # prépare les données x et y pour le train_test_split
        x_papel = x

        x_papel_df = pd.DataFrame(x_papel, columns=['sms'])
        x_papel_vect = self.vectorisation_df(x_papel_df)

        # x et y utilisé pour le train_test_split
        x_new = x_papel_vect.drop('sms', axis=1)

        # résultat de la pipeline
        y_pred = self.model_pip.predict( x_new )
        y_pred_proba = self.model_pip.predict_proba( x_new )

        return y_pred, y_pred_proba