import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.preprocessing import OneHotEncoder


## import data
dataset = pd.read_csv('data/train.csv', header=0)
target = dataset['is_female']

## remove training id and class
dataset.drop(labels=['train_id', 'is_female'], axis=1, inplace =True)
dataset.insert(0, 'is_female', target)


### *** DROP NA SECTION *** ###
dropna = dataset.dropna(axis=1)
print "*** Drop NA ***"

# # summary statistics
# print "Summary"
# print dropna.describe()

# # summary statistic visualization
# print "Variable Boxplots"
# pd.options.display.mpl_style = 'default'
# dropna.iloc[:, range(10)].boxplot()
# plt.show(block=True)

# correlation matrix of variables
corr = dropna.corr()
corr_sorted = corr.sort_values('is_female', ascending=False)
top_ten_corr = corr_sorted[:10]
top_ten_names = top_ten_corr.index

# print "Correlation of Variables to Class"
# print corr_sorted['is_female']
# print top_ten_corr['is_female']

# print "Correlation of all Variables"
# print corr
# print top_ten_corr

# compare male vs. female distributions of top ten variables
top_ten_dropna = dropna[top_ten_names]
top_ten_dropna.groupby('is_female').hist()
plt.show(block=True)

# # correlation matrix visualization
# sns.heatmap(corr,
#              xticklabels=corr.columns.values,
#              yticklabels=corr.columns.values)


# ### FILL NA WITH VALUE SECTION ###
# # replace NA columns with value -1
# dataset_fillna = dataset.fillna(-1)
#
# print "Fill NA Summary"
# print dataset_fillna.describe()
#
#
#
# ### ONE HOT ENCODED SECTION ###
# # one-hot encode categorical variables
# enc = OneHotEncoder()
# enc.fit(dataset_dropna)
# dataset_enc = enc.transform(dataset_dropna)
#
# print "One Hot Encoded Summary"
# print dataset_enc.describe()

