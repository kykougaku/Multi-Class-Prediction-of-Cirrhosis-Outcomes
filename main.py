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
