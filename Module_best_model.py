import pandas as pd
from Module_spam_v3 import model_IA
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Classe definissant la recherche du meilleur modèle
class model_best :

    # Constructeur, avec aucun argument
    def __init__(self, df_json, df_spam, path_csv):
        #ouverture des fichier
        self.df_json = df_json
        self.path_csv = f'{path_csv}\\best_model.csv'
        self.df = df_spam
        
        # liste des modèles
        liste_model = self.df_json['model'].values.tolist()
        self.liste_model = list(set(liste_model))

        #fonction qui initialise les parametre d'un modèle (par default : le modèle est BernoulliNB() )
        self.init_model()
        
        # initialise le data frame des meilleurs modèles
        self.df_best_model = pd.DataFrame(data = {'model': [],
                               'Best_Parameters' : [],
                               'Best_Accuracy' : []})
        
        # fonction qui cherche pour tous les modèles de la listes, les paramètres idéals
        self.grid_serch_best_model()
        

    def init_model (self) :
        # appelle de la classe model avec df_spam et model_commande
        model = BernoulliNB()
        model_spam = model_IA(self.df, model, test=False)
        self.df_prepro = model_spam.df
        self.x_train = model_spam.x_train
        self.y_train = model_spam.y_train
        self.x_test = model_spam.x_test
        self.y_test = model_spam.y_test

    def grid_serch_best_model (self) :
        #pour tous les modèles de la liste :
        for count in range(0,len(self.liste_model)) :

            # choisi le modèle

            if count == 0 :
                model_name = BernoulliNB()
                model = model_IA( self.df, BernoulliNB(), test=False).model_pip
                choix_model = 'BernoulliNB'
                element_pefix = "bernoullinb__"
            if count == 1 :
                model_name = CategoricalNB()
                model = model_IA( self.df, CategoricalNB(), test=False).model_pip
                choix_model = 'CategoricalNB'
                element_pefix = "categoricalnb__"
            if count == 2 :
                model_name = ComplementNB()
                model = model_IA( self.df, ComplementNB(), test=False).model_pip
                choix_model = 'ComplementNB'
                element_pefix = "complementnb__"
            if count == 3 :
                model_name = GaussianNB()
                model = model_IA( self.df, GaussianNB(), test=False).model_pip
                choix_model = 'GaussianNB'
                element_pefix = "gaussiannb__"
            if count == 4 :
                model_name = MultinomialNB()
                model = model_IA( self.df, MultinomialNB(), test=False).model_pip
                choix_model = 'MultinomialNB'
                element_pefix = "multinomialnb__"
            if count == 5 :
                model_name = SVC()
                model = model_IA( self.df, SVC(), test=False).model_pip
                choix_model = 'SVC'
                element_pefix = "svc__"
            if count == 6 :
                model_name = LinearSVC()
                model = model_IA( self.df, LinearSVC(), test=False).model_pip
                choix_model ='LinearSVC'
                element_pefix = "linearsvc__"
            if count == 7 :
                model_name = NuSVC()
                model = model_IA( self.df, NuSVC(), test=False).model_pip
                choix_model = 'NuSVC'
                element_pefix = "nusvc__"
            if count == 8 :
                model_name = NuSVR()
                model = model_IA( self.df, NuSVR(), test=False).model_pip
                choix_model = 'NuSVR'
                element_pefix = "nusvr__"
            if count == 9 :
                model_name = OneClassSVM()
                model = model_IA( self.df, OneClassSVM(), test=False).model_pip
                choix_model = 'OneClassSVM'
                element_pefix = "oneclasssvm__"
            if count == 10 :
                model_name = SVR()
                model = model_IA( self.df, SVR(), test=False).model_pip
                choix_model = 'SVR'
                element_pefix = "svr__"
            if count == 11 :
                model_name = LinearSVR()
                model = model_IA( self.df, LinearSVR(), test=False).model_pip
                choix_model = 'LinearSVR'
                element_pefix = "linearsvr__"
            if count == 12 :
                model_name = KNeighborsClassifier()
                model = model_IA( self.df, KNeighborsClassifier(), test=False).model_pip
                choix_model = 'KNeighborsClassifier'
                element_pefix = "kneighborsclassifier__"
            
            # initialise la liste des parametres de grid_serch pour le modèle
            df_choix = []
            params_str = []
            df_choix = self.df_json.loc[self.df_json['model']==choix_model]
            params_str = df_choix.iloc[0]["param_grid"]
            
            # transforme la liste de parametre (str) en dictionnaire
            df_choix = []
            params_str = []
            df_choix = self.df_json.loc[self.df_json['model']==choix_model]
            params_str = df_choix.iloc[0]["param_grid"]
            
            params = {}
            Best_Parameters_test = []
            for element in params_str :
                value_type = params_str[element].split(",")
                
                value_type_type = []
                for i in value_type :
                    print("i =", i)
                    if i == 'None' :
                        i_type = None
                    if i == 'True' :
                        i_type = True
                    if i == 'False' :
                        i_type = False
                    if i != 'None' and i != 'True' and i != 'False' :
                        try :
                            i_type = int(i)
                        except :
                            try :
                                i_type = float(i)
                            except :
                                i_type = str(i)
                                    
                    print('i_type =',type(i_type))
                    value_type_type.append(i_type)
                
                
                print("element =",element,"element_type = ", type(element))
                print("value =",value_type_type,"value_type = ", type(value_type_type))
                        
                params[f'{element_pefix}{element}'] = value_type_type
                Best_Parameters_test.append(element)
            
            # test : lance la recherche des parametres idéals
            # si succés : l'ajoute à df_best_model l'enregistre dans le fichier best_model
            # si echec : l'ajoute à df_best_model l'enregistre dans le fichier best_model avec comme paramete "error" et comme accuracy 0
            try :
                model_grid = GridSearchCV(model, param_grid=params, n_jobs=-1, cv=5)#, verbose=5
                model_grid.fit(self.x_train,self.y_train)
                
                Best_Parameters = model_grid.best_params_
                Best_Accuracy = model_grid.best_score_
                
                Best_Parameters_save = {}
                for key, value in Best_Parameters.items() :
                    for element in Best_Parameters_test:
                        text = f'{element_pefix}{element}'
                        if key == text :
                            Best_Parameters_save[element] = value
                
                self.df_best_model.loc[count] = [model, Best_Parameters, Best_Accuracy]
                count += 1
                self.df_best_model.to_csv(self.path_csv, index=False)
            
            except :
                
                self.df_best_model.loc[count] = [model, "error", 0]
                count += 1
                self.df_best_model.to_csv(self.path_csv, index=False)

        