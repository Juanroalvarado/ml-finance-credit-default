import statsmodels.formula.api as smf
from pandas import concat

class SplitModel():
    def __init__(self, algo, first_features, rec_features):
        self.first_formula = 'default ~ '+' + '.join(first_features)
        self.rec_formula = 'default ~ '+' + '.join(rec_features)

        self.algo = algo
    
    def train(self, data):
        
        rec_data = data[data['is_first_occurrence']==0]
        first_data = data[data['is_first_occurrence']==1]
        
        self.first_model = self.algo(self.first_formula, data = first_data)
        self.first_fitted_model = self.first_model.fit()

        # if len(rec_data)!=0:
        self.rec_model = self.algo(self.rec_formula, data = rec_data)
        self.rec_fitted_model = self.rec_model.fit()
        print("models fit")

    def predict(self, data):
        rec_data = data[data['is_first_occurrence']==0]
        first_data = data[data['is_first_occurrence']==1]
        print('rec data length',len(rec_data))
        print('first data length',len(first_data))
        
        rec_preds = self.rec_fitted_model.predict(rec_data)
        rec_preds = rec_preds.reindex(rec_data.index)
        
        first_preds = self.first_fitted_model.predict(first_data)
        first_preds = first_preds.reindex(first_data.index)

        predictions = concat([rec_preds,first_preds]).reindex(data.index)
        
        return predictions

    def summary(self):
        print('~~~~~ First Time First Model ~~~~~~')
        print(self.first_fitted_model.summary())
        print(self.first_fitted_model.get_margeff().summary())
        print('\n')
        print('~~~~~ Recurring First Model ~~~~~~')
        print(self.rec_fitted_model.summary())
        print(self.rec_fitted_model.get_margeff().summary())
        print('\n')
        