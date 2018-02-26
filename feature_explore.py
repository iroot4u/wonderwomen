import logging
import time
import pandas as pd

#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


# The functions below only work on Dataframe type objects.
# One hot encoder returns a SparseMatrix type object.
class DataFrame(object):
    def __init__(self, df, title):
      self.df = df
      self.title = title
      self.corr = self.df.corr()
      self.corr_sorted = self.df.corr().sort_values('is_female', ascending=False)

## summary statistics
def summarize(DataFrame):
    print "Summary for %s" % (DataFrame.title)
    print DataFrame.df.describe()

## summary statistic visualization
def boxplotter(DataFrame):
    # col_list = range(10)
    # pd.options.display.mpl_style = 'default'
    # print "Variable Boxplots"
    # DataFrame.df.iloc[:, col_list].boxplot()
    # plt.show(block=True)
    return

# sort correlation matrix based on correlation with 'is_female'
def top_n(DataFrame, n):
    top_n_corr = DataFrame.corr_sorted[:n]
    top_n_names = top_n_corr.index
    print "Correlation of Variables to Class"
    print top_n_corr['is_female']
    return top_n_corr, top_n_names

# compare male vs. female distributions for list of features (ie. top ten features)
def histogram_by_class(DataFrame, feature_list):
    selected_df = DataFrame.df[feature_list]
    selected_df.groupby('is_female').hist()
    plt.show(block=True)
    return

# correlation matrix visualization
def corr_heatmap(corr):
    print corr.shape
    plt.interactive(False)
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    #m.savefig("out.pdf")
    plt.show(block=True)
    time.sleep(100)
    return

def main():

    # load data
    dataset = pd.read_csv('data/train.csv', header=0)
    target = dataset['is_female']

    # remove training id and class
    features = dataset.drop(labels=['train_id', 'is_female'], axis=1, inplace=True)
    dataset.insert(0, 'is_female', target)

    # dropna dataset tests
    dropna = DataFrame(dataset.dropna(axis=1), 'DropNAs')
    summarize(dropna)
    boxplotter(dropna)
    top_10_corr, top_10_names = top_n(dropna, 10)
    histogram_by_class(dropna, top_10_names)
    corr_heatmap(top_10_corr[top_10_names])

if __name__ == "__main__":
    start_time = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s  %(levelname)s:  %(message)s')
    main()
    logging.info("process ran for " + str(time.time() - start_time) + " seconds")
