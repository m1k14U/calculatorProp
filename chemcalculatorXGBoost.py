import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import descriptastorus
from descriptastorus.descriptors import  MakeGenerator

class ChemCalculatorXGRBoost:
    def __init__(self, csv_file):
        self.valid_smiles = []
        self.X = []
        self.generator = MakeGenerator(("RDKit2DNormalized","Morgan3Counts"))
        self.data = pd.read_csv(csv_file)
        self.model = XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror"
                    )

    def show_data(self):
        # Show the dataframe
        print(self.data)

    def select_data(self,number_of_data,feature:str,property:str):
        # Select the number of data and the features (exm. smiles and property)
        self.data = self.data[1:number_of_data]
        self.smiles_data = self.data[feature]
        self.property_data = self.data[property]

    def smiles_to_matrix_descriptors(self):
        # Pass the smiles to descriptors matrix
        if self.smiles_data is not None:
            for smi in self.smiles_data:
                self.descrip = self.generator.process(smi)
                if self.descrip:
                    self.vecto = np.array(self.descrip[1::])
                    self.X.append(self.vecto)
                    self.valid_smiles.append(smi)
            self.X = np.array(self.X)
            return self.X
        else:
            print("Could not be possible to create a matrix. Make sure you select the molecules features with select_data()")

    def input_smiles_to_matrix(self,lisml):
        # The NEW smiles list to predict on a descriptors matrix
        self.X_new = []
        for smi in lisml:
            self.descrip = self.generator.process(smi)
            if self.descrip:
                self.vecto = np.array(self.descrip[1::])
                self.X_new.append(self.vecto)
                self.valid_smiles.append(smi)
        self.X_new = np.array(self.X_new)
        return self.X_new

    def get_shape_matrix_descriptors(self):
        # Show the shape of the 
        print(self.X.shape)

    def prepare_data(self):
        # Select the percentage of train data and define the X_train, y_train, X_test and y_test
        if self.X.any() and self.property_data.any():
            self.y = self.property_data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            return self.X_train, self.X_test, self.y_train, self.y_test
        
    def get_realvalue(self):
        return self.y_test
    
    def get_predvalue(self):
        return self.y_pred

    def train(self):
        # Train the model (X_train, y_train)
        if self.X_train.any() and self.y_train.any() is not None:
            self.model.fit(self.X_train,self.y_train)

            return self.model
    
    def prediction(self):
        # Make a prediction with (X_test) and obtain y_pred 
        if self.X_test.any() and self.y_test.any() is not None:
            self.y_pred = self.model.predict(self.X_test)
            print("MAE: ",mean_absolute_error(self.y_test, self.y_pred))
            print("R2: ", r2_score(self.y_test, self.y_pred))
            
            return self.model
        else:
            print("No data for test!")
    
    def input_prediction(self,smileList):
        # Make a prediction with the NEW smiles list (X_new)
        self.input_smiles_to_matrix(smileList)
        self.y_new = self.model.predict(self.X_new)

        print(self.y_new)

cal = ChemCalculatorXGRBoost("qm9.csv")
cal.select_data(9000,"smiles","gap")
cal.smiles_to_matrix_descriptors()
cal.prepare_data()
cal.train()
cal.prediction()
