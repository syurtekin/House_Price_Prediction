######################################
# House Price Prediction
######################################


from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


######################################
# Exploratory Data Analysis
######################################
# ayrı ayrı ön işlemede yapılabilir
# test ve train birleştirip ön işleme yaptığımızda bilgi sızıntısı olur data leak.(eksiklik ve aykırılıkla çalışırken daha iyi olabilir)
# ikiside tercih edilebilir.
train = pd.read_csv("datasets/house_prices/train.csv")
test = pd.read_csv("datasets/house_prices/test.csv")
df = train.append(test).reset_index(drop=True)
df.head()

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################
# Kategorik Değişken Analizi
##################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

##################
# Sayısal Değişken Analizi
##################

df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot=False)


##################
# Eksik Değerlerin Doldurulması
##################


missing_vs_target(df, "SalePrice", missing_values_table(df, na_name=True))
missing_values_table(df)


none_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
             'GarageArea', 'GarageCars', 'MasVnrArea']
freq_cols = ['Exterior1st', 'Exterior2nd', 'KitchenQual', 'Electrical']

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)
for col in none_cols:
    df[col].replace(np.nan, "None", inplace=True)
for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)

df["Alley"] = df["Alley"].fillna("None")
df["PoolQC"] = df["PoolQC"].fillna("None")
df["MiscFeature"] = df["MiscFeature"].fillna("None")
df["Fence"] = df["Fence"].fillna("None")
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


df.drop(['GarageArea'], axis=1, inplace=True)
df.drop(['GarageYrBlt'], axis=1, inplace=True)
df.drop(['Utilities'], axis=1, inplace=True)

df["GarageCars"] = df["GarageCars"].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[col] = df[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('None')

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))

df["Functional"] = df["Functional"].fillna("Typ")

df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

df['YrSold'] = df['YrSold'].astype(str)


##################
# Target Analizi
##################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

def target_correlation_matrix(dataframe, corr_th=0.5, target="SalePrice"):
    corr = dataframe.corr()
    corr_th = corr_th

    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Yüksek threshold değeri, corr_th değerinizi düşürün!")

target_correlation_matrix(df, corr_th=0.5, target="SalePrice")

######################################
# Data Preprocessing & Feature Engineering
######################################

df.groupby("Neighborhood").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)  # ortalama değerlere göre bir sıralı gruplama yapacağız.

nhood_map = {'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1,'BrkSide': 2, 'Edwards': 2, 'OldTown': 2,'Sawyer': 3, 'Blueste': 3,'SWISU': 4, 'NPkVill': 4, 'NAmes': 4, 'Mitchel': 4,'SawyerW': 5, 'NWAmes': 5,'Gilbert': 6, 'Blmngtn': 6, 'CollgCr': 6,'Crawfor': 7, 'ClearCr': 7,'Somerst': 8, 'Veenker': 8, 'Timber': 8,'StoneBr': 9, 'NridgHt': 9,'NoRidge': 10}

df['Neighborhood'] = df['Neighborhood'].map(nhood_map).astype('int')

df= df.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45", \
50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75", \
80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120", \
150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
"MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", \
7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}})

func = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}
df["Functional"] = df["Functional"].map(func).astype("int")
df.groupby("Functional").agg({"SalePrice": "mean"})

# MSZoning
df.loc[(df["MSZoning"] == "C (all)"), "MSZoning"] = 1
df.loc[(df["MSZoning"] == "RM"), "MSZoning"] = 2
df.loc[(df["MSZoning"] == "RH"), "MSZoning"] = 2
df.loc[(df["MSZoning"] == "RL"), "MSZoning"] = 3
df.loc[(df["MSZoning"] == "FV"), "MSZoning"] = 3
# LotShape
df.groupby("LotShape").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
shape_map = {"Reg": 1, "IR1": 2, "IR3": 3, "IR2": 4}
df['LotShape'] = df['LotShape'].map(shape_map).astype('int')
# LandContour
df.groupby("LandContour").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
contour_map = {"Bnk": 1, "Lvl": 2, "Low": 3, "HLS": 4}
df['LandContour'] = df['LandContour'].map(contour_map).astype('int')

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "SalePrice", cat_cols)

# LotConfig
df.loc[(df["LotConfig"] == "Inside"), "LotConfig"] = 1
df.loc[(df["LotConfig"] == "FR2"), "LotConfig"] = 1
df.loc[(df["LotConfig"] == "Corner"), "LotConfig"] = 1
df.loc[(df["LotConfig"] == "FR3"), "LotConfig"] = 2
df.loc[(df["LotConfig"] == "CulDSac"), "LotConfig"] = 2

# Condition1
cond1_map = {"Artery": 1, "RRAe": 1, "Feedr": 1,"Norm": 2, "RRAn": 2, "RRNe": 2,"PosN": 3, "RRNn": 3, "PosA": 3}
df['Condition1'] = df['Condition1'].map(cond1_map).astype('int')


# BldgType
df.loc[(df["BldgType"] == "2fmCon"), "BldgType"] = 1
df.loc[(df["BldgType"] == "Duplex"), "BldgType"] = 1
df.loc[(df["BldgType"] == "Twnhs"), "BldgType"] = 1
df.loc[(df["BldgType"] == "1Fam"), "BldgType"] = 2
df.loc[(df["BldgType"] == "TwnhsE"), "BldgType"] = 2

# RoofStyle
df.groupby("RoofStyle").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
df.loc[(df["RoofStyle"] == "Gambrel"), "RoofStyle"] = 1
df.loc[(df["RoofStyle"] == "Gablee"), "RoofStyle"] = 2
df.loc[(df["RoofStyle"] == "Mansard"), "RoofStyle"] = 3
df.loc[(df["RoofStyle"] == "Flat"), "RoofStyle"] = 4
df.loc[(df["RoofStyle"] == "Hip"), "RoofStyle"] = 5
df.loc[(df["RoofStyle"] == "Shed"), "RoofStyle"] = 6

# RoofMatl
df.groupby("RoofMatl").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
df.loc[(df["RoofMatl"] == "Roll"), "RoofMatl"] = 1
df.loc[(df["RoofMatl"] == "ClyTile"), "RoofMatl"] = 2
df.loc[(df["RoofMatl"] == "CompShg"), "RoofMatl"] = 3
df.loc[(df["RoofMatl"] == "Metal"), "RoofMatl"] = 3
df.loc[(df["RoofMatl"] == "Tar&Grv"), "RoofMatl"] = 3
df.loc[(df["RoofMatl"] == "WdShake"), "RoofMatl"] = 4
df.loc[(df["RoofMatl"] == "Membran"), "RoofMatl"] = 4
df.loc[(df["RoofMatl"] == "WdShngl"), "RoofMatl"] = 5

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "SalePrice", cat_cols)

# ExterQual
df.groupby("ExterQual").agg({"SalePrice": "mean"}).sort_values(by="SalePrice", ascending=False)
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterQual'] = df['ExterQual'].map(ext_map).astype('int')

# ExterCond
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterCond'] = df['ExterCond'].map(ext_map).astype('int')

# BsmtQual
bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtQual'] = df['BsmtQual'].map(bsm_map).astype('int')

# BsmtCond
bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtCond'] = df['BsmtCond'].map(bsm_map).astype('int')

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "SalePrice", cat_cols)

# BsmtFinType1
bsm_map = {'None': 0, 'Rec': 1, 'BLQ': 1, 'LwQ': 2, 'ALQ': 3, 'Unf': 3, 'GLQ': 4}
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsm_map).astype('int')

# BsmtFinType2
bsm_map = {'None': 0, 'BLQ': 1, 'Rec': 2, 'LwQ': 2, 'Unf': 3, 'GLQ': 3, 'ALQ': 4}
df['BsmtFinType2'] = df['BsmtFinType2'].map(bsm_map).astype('int')

# BsmtExposure
bsm_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
df['BsmtExposure'] =df['BsmtExposure'].map(bsm_map).astype('int')

# Heating
heat_map = {'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5}
df['Heating'] = df['Heating'].map(heat_map).astype('int')

# HeatingQC
heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['HeatingQC'] = df['HeatingQC'].map(heat_map).astype('int')

# KitchenQual
kitch_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['KitchenQual'] = df['KitchenQual'].map(heat_map).astype('int')

# FireplaceQu
fire_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['FireplaceQu'] = df['FireplaceQu'].map(fire_map).astype('int')

# GarageCond
garage_map = {'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['GarageCond'] = df['GarageCond'].map(garage_map).astype('int')

# GarageQual
garage_map = {'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Ex': 4, 'Gd': 5}
df['GarageQual'] = df['GarageQual'].map(garage_map).astype('int')

# PavedDrive
paved_map = {'N': 1, 'P': 2, 'Y': 3}
df['PavedDrive'] = df['PavedDrive'].map(paved_map).astype('int')

# CentralAir
cent = {"N": 0, "Y": 1}
df["CentralAir"] = df["CentralAir"].map(cent).astype("int")
df.groupby("CentralAir").agg({"SalePrice": "mean"})

# LandSlope
df.loc[df["LandSlope"] == "Gtl", "LandSlope"] = 1
df.loc[df["LandSlope"] == "Sev", "LandSlope"] = 2
df.loc[df["LandSlope"] == "Mod", "LandSlope"] = 2
df["LandSlope"] = df["LandSlope"].astype("int")

# OverallQual
df.loc[df["OverallQual"] == 1, "OverallQual"] = 1
df.loc[df["OverallQual"] == 2, "OverallQual"] = 1
df.loc[df["OverallQual"] == 3, "OverallQual"] = 1
df.loc[df["OverallQual"] == 4, "OverallQual"] = 2
df.loc[df["OverallQual"] == 5, "OverallQual"] = 3
df.loc[df["OverallQual"] == 6, "OverallQual"] = 4
df.loc[df["OverallQual"] == 7, "OverallQual"] = 5
df.loc[df["OverallQual"] == 8, "OverallQual"] = 6
df.loc[df["OverallQual"] == 9, "OverallQual"] = 7
df.loc[df["OverallQual"] == 10, "OverallQual"] = 8

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "SalePrice", cat_cols)

df.head()

df["NEW"] = df["GarageCars"] * df["OverallQual"]
df["NEW3"] = df["TotalBsmtSF"] * df["1stFlrSF"]
df["NEW4"] = df["TotRmsAbvGrd"] * df["GrLivArea"]
df["NEW5"] = df["FullBath"] * df["GrLivArea"]
df["NEW6"] = df["YearBuilt"] * df["YearRemodAdd"]
df["NEW7"] = df["OverallQual"] * df["YearBuilt"]
df["NEW8"] = df["OverallQual"] * df["RoofMatl"]
df["NEW9"] = df["PoolQC"] * df["OverallCond"]
df["NEW10"] = df["OverallCond"] * df["MasVnrArea"]
df["NEW11"]  = df["LotArea"] * df["GrLivArea"]
df["NEW12"] = df["FullBath"] * df["GrLivArea"]
df["NEW13"] = df["FullBath"] * df["TotRmsAbvGrd"]
df["NEW14"] = df["1stFlrSF"] *df["TotalBsmtSF"]
df["New_Home_Quality"] =  df["OverallCond"] / df["OverallQual"]
df['POOL'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['HAS2NDFLOOR'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df["LUXURY"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["New_TotalBsmtSFRate"] = df["TotalBsmtSF"] / df["LotArea"]
df['TotalPorchArea'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
df['IsNew'] = df.YearBuilt.apply(lambda x: 1 if x > 2000 else 0)
df['IsOld'] = df.YearBuilt.apply(lambda x: 1 if x < 1946 else 0)

##################
# Rare Encoding
##################

rare_analyser(df, "SalePrice", cat_cols)

df = rare_encoder(df, 0.01, cat_cols)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]


cat_cols = [col for col in cat_cols if col not in useless_cols]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", cat_cols)

##################
# Label Encoding & One-Hot Encodıng
##################


cat_cols = cat_cols + cat_but_car


df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)


useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

df[useless_cols_new].head()


for col in useless_cols_new:
    cat_summary(df, col)


rare_analyser(df, "SalePrice", useless_cols_new)


##################
# Missing Values
##################
missing_values_table(df)

test.shape

missing_values_table(train)


na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

df.isnull().sum().sum()
##################
# Outliers
##################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)


######################################
# Modeling
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

##################
# Base Models
##################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

##################
# Hyperparameter Optimization
##################

lgbm_model = LGBMRegressor(random_state=46)

# modelleme öncesi hata:
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=10, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1, 0.03, 0.2, 0.5],
               "n_estimators": [100, 200, 250, 500, 1500],
               "colsample_bytree": [0.3,0.4, 0.5, 0.7, 1]}

#parametre değerleri için ön tanımlı değerlerine bak. onun etrafında gez.

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model_lgbm = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model_lgbm, X, y, cv=10, scoring="neg_mean_squared_error")))

# 0.1209

# CatBoost

catboost_model = CatBoostRegressor(random_state = 46)

catboost_params = {"iterations": [200, 250, 300, 500],
                   "learning_rate": [0.01, 0.1, 0.2, 0.5],
                   "depth": [3, 6]}

rmse = np.mean(np.sqrt(-cross_val_score(catboost_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))


cat_gs_best = GridSearchCV(catboost_model,
catboost_params,cv=3,n_jobs=-1,verbose=True).fit(X, y)

final_model_cat = catboost_model.set_params(**cat_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_cat, X, y, cv=10, scoring="neg_mean_squared_error")))

# 0.1198

# GBM

gbm_model = GradientBoostingRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))

gbm_params = {"learning_rate": [0.01,0.05,0.1],"max_depth": [3,5,8],"n_estimators": [500,1000,1500],"subsample": [1, 0.5, 0.7]}

gbm_gs_best = GridSearchCV(gbm_model,gbm_params,cv=5,n_jobs=-1,verbose=True).fit(X, y)

final_model_gbm = gbm_model.set_params(**gbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_gbm, X, y, cv=10, scoring="neg_mean_squared_error")))

# 0.1190
