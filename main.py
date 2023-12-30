import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder,OrdinalEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

def get_data():
    train = pd.read_csv('train.csv')
    train = train.drop(['id'], axis=1)
    test = pd.read_csv('test.csv')
    original = pd.read_csv('cirrhosis.csv')
    original = original.drop(['ID'], axis=1)
    return train, test, original

def print_df(df,name):
    print("{0} rows:{1} columns:{2}".format(name, df.shape[0], df.shape[1]))
    print(df.head(5))
    print("\n")

if __name__ == '__main__':
    train , test, original = get_data()

    print_df(train,'train')
    print_df(test,'test')
    print_df(original,'original')

    testid = test['id']

    originalstatus = original['Status']
    original = original.drop(['Status'], axis=1)
    original = pd.concat([original, originalstatus], axis=1)

    train = pd.concat([train, original], axis=0)

    target = 'Status'
    label_features = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
    num_features = [x for x in train.columns if x not in label_features + [target]]

    train = train.dropna()

    labelendoer = LabelEncoder()
    train[target] = labelendoer.fit_transform(train[target])

    encoders = {
    'Drug': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[['Placebo', 'D-penicillamine']]),
    'Sex': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Ascites': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Hepatomegaly': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Spiders': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Edema': OneHotEncoder(),
    'Stage': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    }

    for feat, enc in encoders.items():
        if isinstance(enc, OrdinalEncoder):
            train[feat] = enc.fit_transform(train[[feat]]).astype('int32')
            test[feat] = enc.transform(test[[feat]]).astype('int32')
        if isinstance(enc, OneHotEncoder):
            new_cols = enc.fit_transform(train[[feat]]).toarray().astype('int8')
            col_names = enc.get_feature_names_out()
            
            train[col_names] = new_cols
            train.drop(feat, axis=1, inplace=True)
            
            new_cols_test = enc.transform(test[[feat]]).toarray().astype('int8')
            test[col_names] = new_cols_test
            test.drop(feat, axis=1, inplace=True)
    




    class DiagnosisDateTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X['Diagnosis_Date'] = X['Age'] - X['N_Days']
            return X
        
    class AgeYearsTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X['Age_Years'] = round(X['Age'] / 365.25).astype("int16")
            return X

    class AgeGroupsTransformer(BaseEstimator, TransformerMixin):
        """Older people might be hit harder (interaction) by health issues. Also can cover lifestyle influences, i.e.
        alcohol consumption etc."""
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            # Use years from above, min=26, max=78
            X['Age_Group'] = pd.cut(X['Age_Years'], bins=[19, 29, 49, 64, 99], labels = [0, 1, 2, 3]).astype('int16')
            return X

    class BilirubinAlbuminTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X['Bilirubin_Albumin'] = X['Bilirubin'] * X['Albumin']
            return X

    class DrugEffectivenessTransformer(BaseEstimator, TransformerMixin):
        # Placeholder concept, assuming 'Bilirubin' improvement is a measure of effectiveness
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X['Drug_Effectiveness'] = X['Drug'] * X['Bilirubin']
            return X

    class SymptomScoreTransformer(BaseEstimator, TransformerMixin):
        # From data set explanations above let's add all the "bad" symptoms
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            # symptom_columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
            symptom_columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema_N', 'Edema_S', 'Edema_Y']
            X['Symptom_Score'] = X[symptom_columns].sum(axis=1)
            return X
        
    class SymptomCatTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.symptom_columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema_N', 'Edema_S', 'Edema_Y']
            self.encoder = OneHotEncoder(handle_unknown='ignore')

        def fit(self, X, y=None):
            X_copy = X.copy()
            symptom_scores = X_copy[self.symptom_columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
            self.encoder.fit(symptom_scores.values.reshape(-1, 1))
            return self

        def transform(self, X):
            X_transformed = X.copy()
            symptom_scores = X_transformed[self.symptom_columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
            
            encoded_features = self.encoder.transform(symptom_scores.values.reshape(-1, 1)).toarray().astype("int8")
            encoded_feature_names = self.encoder.get_feature_names_out(input_features=['Symptom_Score'])

            # Drop the original symptom columns and add the new encoded features
            # X_transformed.drop(columns=self.symptom_columns, inplace=True)
            X_transformed[encoded_feature_names] = pd.DataFrame(encoded_features, index=X_transformed.index)
            
            return X_transformed

    class LiverFunctionTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            liver_columns = ['Bilirubin', 'Albumin', 'Alk_Phos', 'SGOT']
            X['Liver_Function_Index'] = X[liver_columns].mean(axis=1)
            return X

    class RiskScoreTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X['Risk_Score'] = X['Bilirubin'] + X['Albumin'] - X['Alk_Phos']
            return X

    class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X['Diag_Year'] = (X['N_Days'] / 365).astype(int)
            X['Diag_Month'] = ((X['N_Days'] % 365) / 30).astype(int)
            return X
        
    class ScalingTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler = StandardScaler()
            self.num_feats = num_features + ['Diagnosis_Date', 'Age_Years', 'Bilirubin_Albumin', 'Drug_Effectiveness', 
                                        'Symptom_Score', 'Liver_Function_Index', 'Risk_Score', 'Diag_Year', 'Diag_Month']

        def fit(self, X, y=None):
            self.scaler.fit(X[self.num_feats])
            return self

        def transform(self, X):
            X_scaled = X.copy()
            X_scaled[self.num_feats] = self.scaler.transform(X_scaled[self.num_feats])
            return X_scaled

    # Define the pipeline
    pipeline = Pipeline([
        ('diagnosis_date', DiagnosisDateTransformer()),
        ('age_years', AgeYearsTransformer()),
        ('age_groups', AgeGroupsTransformer()),
        ('bilirubin_albumin', BilirubinAlbuminTransformer()),
        ('drug_effectiveness', DrugEffectivenessTransformer()),
        ('symptom_score', SymptomScoreTransformer()),
        ('symptom_cat_score', SymptomCatTransformer()),
        ('liver_function', LiverFunctionTransformer()),
        ('risk_score', RiskScoreTransformer()),
        ('time_features', TimeFeaturesTransformer()),
        #('scaling', ScalingTransformer()),
        # ... ?
    ])

    # Apply the pipeline to your dataframes
    train = pipeline.fit_transform(train)
    test = pipeline.transform(test)

    # Update the CAT_FEATS
    CAT_FEATS = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage', #old
                'Age_Group', 'Symptom_Score'] # new 
    




    tmp_df = train.copy()

    # Calculate the mean and standard deviation for each column
    means = tmp_df[num_features].mean()
    std_devs = tmp_df[num_features].std()

    # Define a threshold for what you consider to be an outlier, typically 3 standard deviations from the mean
    n_stds = 6
    thresholds = n_stds * std_devs

    # Detect outliers
    outliers = (np.abs(tmp_df[num_features] - means) > thresholds).any(axis=1)

    print(f"Detected {sum(outliers)} that are more than {n_stds} SDs away from mean...")

    # The resulting boolean series can be used to filter out the outliers
    outliers_df = tmp_df[outliers]

    # Overwrite the train data
    train = tmp_df[~outliers].reset_index(drop=True)




    y = train[target]
    X = train.drop([target], axis=1)

    my_pipeline = Pipeline(steps=[("model", lgb.LGBMClassifier(n_estimators=900, learning_rate=0.006,n_jobs=4,random_state=0,num_class=3,objective='multiclass'))])
    score = cross_val_score(my_pipeline, X, y, cv=2, scoring='neg_log_loss').mean()
    print(score)

    model = lgb.LGBMClassifier(n_estimators=900, learning_rate=0.006,n_jobs=4,random_state=0,num_class=3,objective='multiclass')
    model.fit(X, y)

    y_test_hat = model.predict_proba(test[X.columns])
    assert y_test_hat.shape == (test.shape[0], 3)
    submission_labels = ["Status_C", "Status_CL", "Status_D"]

    sub = pd.DataFrame(
        {"id": testid, **dict(zip(submission_labels, y_test_hat.T))}
    )

    sub.to_csv('submission.csv', index=False)
