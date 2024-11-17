import statsmodels.formula.api as smf
from pandas import concat, Series


class SplitModel():
    def __init__(self, params, first_features, rec_features):
        self.first_features = first_features
        self.rec_features = rec_features

        self.params = params
    
    def train(self, data):
        
        rec_data = data[data['is_first_occurrence']==0]
        first_data = data[data['is_first_occurrence']==1]
        

        self.rec_model = XGBClassifier(**self.params)
        self.first_model = XGBClassifier(**self.params)
        
        self.first_fitted_model = self.first_model.fit(X=first_data[self.first_features], 
                   y=first_data['default'])
        self.rec_fitted_model = self.rec_model.fit(X=rec_data[self.rec_features], 
                   y=rec_data['default'])
        
        print("models fitted")

    def predict(self, data):
        data['prediction_index'] = range(0, len(data))
        rec_data = data[data['is_first_occurrence']==0]
        first_data = data[data['is_first_occurrence']==1]
        print('rec data length',len(rec_data))
        print('first data length',len(first_data))
        
        rec_preds = Series(self.rec_fitted_model.predict_proba(rec_data[self.rec_features])[:,1])
        rec_preds.index = rec_data.prediction_index
        
        first_preds = Series(self.first_fitted_model.predict_proba(first_data[self.first_features])[:,1])
        first_preds.index = first_data.prediction_index

        predictions = concat([rec_preds,first_preds]).reindex(data.prediction_index)
        
        return predictions

    # def summary(self):
    #     print('~~~~~ First Time First Model ~~~~~~')
    #     print(self.first_fitted_model.summary())
    #     print(self.first_fitted_model.get_margeff().summary())
    #     print('\n')
    #     print('~~~~~ Recurring First Model ~~~~~~')
    #     print(self.rec_fitted_model.summary())
    #     print(self.rec_fitted_model.get_margeff().summary())
    #     print('\n')