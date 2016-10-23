
import sys
import time
import h5py
import codecs
import subprocess
import numpy as np
import pandas as pd
import skfuzzy as fuzz

from argparse import ArgumentParser
# from fnmtf import fnmtf
from davies_bouldin import davies_bouldin_score, calculate_centroids_doc_mean
# from onmtf import matrix_factorization_clustering
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class Dataset(object):

    @classmethod
    def fromdataframe(cls, dataframe):
        s = cls()
        s.__dataframe = dataframe
        s.__target_names = dataframe['channel'].unique().tolist()
        s.__target = s.build_targets()
        s.__data = dataframe['all']
        s.__name = 'ig'
        return s

    @classmethod
    def fromnumpyarray(cls, arr, labels):
        s = cls()
        s.__target = labels
        s.__data = arr
        s.__name = 'nips'
        return s

    def build_targets(self):
        classes_index = list(range(len(self.target_names)))
        target_names_to_index = {k: v for k, v in zip(self.target_names, classes_index)}

        return np.array([target_names_to_index[name] for name in self.dataframe['channel']])

    @property
    def name(self):
        return self.__name

    @property
    def dataframe(self):
        return self.__dataframe

    @property
    def target(self):
        return self.__target

    @property
    def target_names(self):
        return self.__target_names

    @property
    def data(self):
        return self.__data


def get_dataset(dataset_name):
    if dataset_name == 'newsgroup':
        return fetch_20newsgroups(subset='all')
    elif dataset_name == 'ig':
        ig_df = pd.read_pickle('all_news_df.pkl')
        return Dataset.fromdataframe(ig_df)
    elif dataset_name == 'igtoy':
        arena_news_df = pd.read_pickle('arena_news_df.pkl')
        sport_news_df = pd.read_pickle('sport_news_df.pkl')
        jovem_news_df = pd.read_pickle('jovem_news_df.pkl')
        labels_true = np.array(len(arena_news_df.ix[0:99])*[0] + len(sport_news_df.ix[0:99])*[1] + len(jovem_news_df.ix[0:99])*[2])
        count_vect = CountVectorizer(encoding='UTF-8',lowercase=False, min_df=2)
        X = count_vect.fit_transform(arena_news_df['all'].ix[0:99].tolist() + sport_news_df['all'].ix[0:99].tolist() + jovem_news_df['all'].ix[0:99].tolist())
        return Dataset.fromnumpyarray(X, labels_true)
    elif dataset_name == 'nips':
        arr = np.load('nips_data')
        labels = np.load('nips_labels')
        return Dataset.fromnumpyarray(arr, labels)


def preprocess(dataset):
    if dataset.name == 'ig':
        vectorizer = CountVectorizer(stop_words='english', min_df=2)
        X = vectorizer.fit_transform(dataset.data)
    else:
        X = dataset.data

    X_train_norm_tfidf = TfidfTransformer(norm=u'l2', use_idf=True).fit_transform(X)
    X_train_tfidf = TfidfTransformer(use_idf=True).fit_transform(X)
    X_train_norm = TfidfTransformer(norm=u'l2', use_idf=False).fit_transform(X)
    X_train = TfidfTransformer(use_idf=False).fit_transform(X)

    return X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf


def run_kmeans(X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf, labels_true, dataset_name, kk, ll):
    params = {
        'newsgroup': {
            'k': [10, 15, 20, 25, 30],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'ig': {
            'k': [13],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'igtoy': {
            'k': [3],
            'l': [2, 3, 4, 5, 6],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'nips': {
            'k': [9],
            'l': [5, 7, 9, 11, 13],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        }
    }
    output_file = codecs.open(dataset_name + '_kmeans_news_results.csv', 'w', 'utf-8')
    output_file.write('X,K,NMI,RAND,DAVIES\n')
    for k in params[dataset_name]['k']:
        for data_str in params[dataset_name]['X']:
            data = eval(data_str)
            data = data.toarray().astype(np.float64)

            error_best = np.inf
            for _ in range(10):
                tick1 = time.time()
                datat = data.T
                # n, _ = data.shape
                # temp = np.diag(np.squeeze(np.asarray((data.dot(datat).dot(np.ones(n).reshape(n, 1))))))
                # d = datat.dot(np.sqrt(temp))
                estimator = KMeans(n_clusters=k, max_iter=10000)
                estimator.fit(data)
                tick2 = time.time()
                print(u'Took {} secs to train the {} model...'.format((tick2 - tick1), 'kmeans'))

                labels_pred = estimator.labels_
                centroids = estimator.cluster_centers_
                error = estimator.inertia_

                nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
                rand_score = adjusted_rand_score(labels_true, labels_pred)
                davies_score = davies_bouldin_score(data, labels_pred, centroids)
                tick3 = time.time()
                print(u'Took {} secs to calculate {} metrics...'.format((tick3 - tick2), 'kmeans'))

                output_file.write(u'{},{},{},{},{}\n'.format(data_str, k, nmi_score, rand_score, davies_score))

            print('Execution: X: {}, k: {}'.format(data_str, k))
            print('NMI score: {}'.format(nmi_score))
            print('Rand score: {}'.format(rand_score))
            print('Davies score: {}'.format(davies_score))
            print('-----------------------------------------------\n')

    output_file.close()


def run_fkmeans(X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf, labels_true, dataset_name, kk, ll):
    params = {
        'newsgroup': {
            'k': [20],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'ig': {
            'k': [13],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'igtoy': {
            'k': [3],
            'l': [2, 3, 4, 5, 6],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'nips': {
            'k': [9],
            'l': [5, 7, 9, 11, 13],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        }
    }
    output_file = codecs.open(dataset_name + '_fuzzy_cmeans_news_results.csv', 'w', 'utf-8')
    output_file.write('X,K,NMI,RAND,DAVIES\n')
    output_file.flush()
    for k in params[dataset_name]['k']:
        for data_str in params[dataset_name]['X']:
            data = eval(data_str)
            data = data.toarray().astype(np.float64)

            error_best = np.inf
            for _ in range(10):
                tick1 = time.time()
                centroids, U, _, _, errors, _, _ = fuzz.cluster.cmeans(
                    data.T,
                    k,
                    2,
                    error=0.00000000001,
                    maxiter=10000)
                tick2 = time.time()
                print(u'Took {} secs to train the {} model...'.format((tick2 - tick1), 'fkmeans'))

                labels_pred = np.argmax(U, axis=0)
                error = errors[-1]

                nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
                rand_score = adjusted_rand_score(labels_true, labels_pred)
                davies_score = davies_bouldin_score(data, labels_pred, centroids)
                tick3 = time.time()
                print(u'Took {} secs to calculate {} metrics...'.format((tick3 - tick2), 'fkmeans'))

                output_file.write(u'{},{},{},{},{}\n'.format(data_str, k, nmi_score, rand_score, davies_score))
                output_file.flush()

                print('Execution: X: {}, k: {}'.format(data_str, k))
                print('NMI score: {}'.format(nmi_score))
                print('Rand score: {}'.format(rand_score))
                print('Davies score: {}'.format(davies_score))
                print('-----------------------------------------------\n')

    output_file.close()


def run_onmtf(X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf, labels_true, dataset_name, kk, ll):
    params = {
        'newsgroup': {
            'k' : [20],
            'l' : [15, 20, 25, 30],
            'X' : ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'igtoy': {
            'k': [3],
            'l': [2, 3, 4, 5, 6],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'ig': {
            'k' : [7, 10, 13, 16, 19],
            'l' : [19],
            'X' : ['X_train_norm_tfidf']
            # 'X' : ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'nips': {
            'k': [9],
            'l': [6, 9, 12, 15, 18],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        }
    }
    if kk:
        filename = dataset_name + '_kk=' + str(kk) + '_ll=' + str(ll) + '_onmtf_news_results.csv'
        params[dataset_name]['k'] = [int(kk)]
        params[dataset_name]['l'] = [int(ll)]
    else:
        filename = dataset_name + '_onmtf_news_results.csv'

    out_f = codecs.open(filename, 'w', 'utf-8')
    out_f.write('X,K,L,NMI,RAND,DAVIES\n')
    for k in params[dataset_name]['k']:
        for l in params[dataset_name]['l']:
            for data_str in params[dataset_name]['X']:
                data = eval(data_str)
                data = data.toarray().astype(np.float64)

                h5f = h5py.File('data.h5', 'w')
                h5f.create_dataset('X', data=data.T)
                h5f.close()

                error_best = np.inf
                for _ in range(10):
                    tick1 = time.time()
                    proc = subprocess.Popen(['./algos_gpu', 'onmtf', str(k), str(l), '10000'],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

                    (out, err) = proc.communicate()
                    print 'out:', out

                    U = np.genfromtxt('U.csv', delimiter=',')
                    S = np.genfromtxt('S.csv', delimiter=',')
                    V = np.genfromtxt('V.csv', delimiter=',')
                    with open('error.csv') as f:
                        error = float(f.read())
                    labels_pred = np.argmax(U, axis=1)
                    tick2 = time.time()
                    print(u'Took {} secs to train the {} model...'.format((tick2 - tick1), 'onmtf'))

                    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
                    rand_score = adjusted_rand_score(labels_true, labels_pred)
                    davies_score = davies_bouldin_score(data, labels_pred, calculate_centroids_doc_mean(data, labels_pred, k))
                    tick3 = time.time()
                    print(u'Took {} secs to calculate {} metrics...'.format((tick3 - tick2), 'onmtf'))


                    out_f.write(u'{},{},{},{},{},{}\n'.format(data_str, k, l, nmi_score, rand_score, davies_score))

                    print('Execution: X: {}, k: {}'.format(data_str, k))
                    print('Algo error: {}'.format(error_best))
                    print('NMI score: {}'.format(nmi_score))
                    print('Rand score: {}'.format(rand_score))
                    print('Davies score: {}'.format(davies_score))
                    print('-----------------------------------------------\n')
    out_f.close()


def run_fnmtf(X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf, labels_true, dataset_name, kk, ll):
    params = {
        'newsgroup': {
            'k': [20],
            'l': [15, 20, 25, 30],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'igtoy': {
            'k': [3],
            'l': [2, 3, 4, 5, 6],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'ig': {
            'k': [7, 10, 13, 16, 19],
            'l': [7, 10, 13, 16, 19],
            # 'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'nips': {
            'k': [9],
            'l': [6, 9, 12, 15, 18],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
            # 'X': ['X_train', 'X_train_tfidf']
        }
    }

    if kk:
        filename = dataset_name + '_kk=' + str(kk) + '_ll=' + str(ll) + '_fnmtf_news_results.csv'
        params[dataset_name]['k'] = [int(kk)]
        params[dataset_name]['l'] = [int(ll)]
    else:
        filename = dataset_name + '_fnmtf_news_results.csv'

    out_f = codecs.open(filename, 'w', 'utf-8')
    out_f.write('X,K,L,NMI,RAND,DAVIES\n')
    for k in params[dataset_name]['k']:
        for l in params[dataset_name]['l']:
            for data_str in params[dataset_name]['X']:
                data = eval(data_str)
                data = data.toarray().astype(np.float64)

                h5f = h5py.File('data.h5', 'w')
                h5f.create_dataset('X', data=data.T)
                h5f.close()

                error_best = np.inf
                for _ in xrange(10):
                    tick1 = time.time()
                    # U, S, V, labels_pred, _, error = fnmtf(data, k, l)

                    proc = subprocess.Popen(['./algos', 'fnmtf', str(k), str(l), '10000'],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

                    (out, err) = proc.communicate()
                    print('out: {}'.format(out))

                    U = np.genfromtxt('U.csv', delimiter=',')
                    S = np.genfromtxt('S.csv', delimiter=',')
                    V = np.genfromtxt('V.csv', delimiter=',')
                    with open('error.csv') as f:
                        error = float(f.read())
                    labels_pred = np.argmax(U, axis=1)

                    tick2 = time.time()
                    print(u'Took {} secs to train the {} model...'.format((tick2 - tick1), 'fnmtf'))

                    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
                    rand_score = adjusted_rand_score(labels_true, labels_pred)
                    davies_score = davies_bouldin_score(data, labels_pred, calculate_centroids_doc_mean(data, labels_pred, k))

                    out_f.write(u'{},{},{},{},{},{}\n'.format(data_str, k, l, nmi_score, rand_score, davies_score))

                    print('Execution: X: {}, k: {}, l: {}'.format(data_str, k, l))
                    print('Algo error: {}'.format(error_best))
                    print('NMI score: {}'.format(nmi_score))
                    print('Rand score: {}'.format(rand_score))
                    print('Davies score: {}'.format(davies_score))
                    print('-----------------------------------------------\n')


def run_ovnmtf(X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf, labels_true, dataset_name, kk, ll):
    params = {
        'newsgroup': {
            'k': [20],
            'l': [15, 20, 25, 30],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'igtoy': {
            'k': [3],
            'l': [2, 3, 4, 5, 6],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'ig': {
            'k': [13],
            'l': [7, 10, 13, 16, 19],
            # 'X': ['X_train_norm_tfidf']
            'X': ['X_train_norm', 'X_train_tfidf']
        },
        'nips': {
            'k': [9],
            'l': [6, 9, 12, 15, 18],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        }
    }

    if kk:
        filename = dataset_name + '_kk=' + str(kk) + '_ll=' + str(ll) + '_ovnmtf_news_results.csv'
        params[dataset_name]['k'] = [int(kk)]
        params[dataset_name]['l'] = [int(ll)]
    else:
        filename = dataset_name + '_ovnmtf_news_results.csv'

    out_f = codecs.open(filename, 'w', 'utf-8')
    out_f.write('X,K,L,NMI,RAND,DAVIES\n')
    for k in params[dataset_name]['k']:
        for l in params[dataset_name]['l']:
            for data_str in params[dataset_name]['X']:
                data = eval(data_str)
                data = data.toarray().astype(np.float64)

                h5f = h5py.File('data.h5', 'w')
                h5f.create_dataset('X', data=data.T)
                h5f.close()

                error_best = np.inf
                for _ in xrange(10):
                    tick1 = time.time()
                    # U, S, V, labels_pred, _, error = fnmtf(data, k, l)

                    proc = subprocess.Popen(['./algos_gpu', 'ovnmtf', str(k), str(l), '10000'],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

                    (out, err) = proc.communicate()
                    print('out: {}'.format(out))

                    U = np.genfromtxt('U.csv', delimiter=',')
                    S = np.genfromtxt('S.csv', delimiter=',')
                    # V = np.genfromtxt('V.csv', delimiter=',')
                    with open('error.csv') as f:
                        error = float(f.read())
                    labels_pred = np.argmax(U, axis=1)

                    tick2 = time.time()
                    print(u'Took {} secs to train the {} model...'.format((tick2 - tick1), 'ovnmtf'))

                    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
                    rand_score = adjusted_rand_score(labels_true, labels_pred)
                    davies_score = davies_bouldin_score(data, labels_pred, calculate_centroids_doc_mean(data, labels_pred, k))

                    out_f.write(u'{},{},{},{},{},{}\n'.format(data_str, k, l, nmi_score, rand_score, davies_score))

                    print('Execution: X: {}, k: {}, l: {}'.format(data_str, k, l))
                    print('Algo error: {}'.format(error_best))
                    print('NMI score: {}'.format(nmi_score))
                    print('Rand score: {}'.format(rand_score))
                    print('Davies score: {}'.format(davies_score))
                    print('-----------------------------------------------\n')



def run_bin_ovnmtf(X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf, labels_true, dataset_name, kk, ll):
    params = {
        'newsgroup': {
            'k': [20],
            'l': [15, 20, 25, 30],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'igtoy': {
            'k': [3],
            'l': [2, 3, 4, 5, 6],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'ig': {
            'k': [7, 10, 13, 16, 19],
            'l': [7, 10, 13, 16, 19],
            'X': ['X_train_norm_tfidf']
            # 'X': ['X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        },
        'nips': {
            'k': [9],
            'l': [6, 9, 12, 15, 18],
            'X': ['X_train', 'X_train_norm', 'X_train_tfidf', 'X_train_norm_tfidf']
        }
    }

    if kk:
        filename = dataset_name + '_kk=' + str(kk) + '_ll=' + str(ll) + '_X=' + params[dataset_name]['X'][0] + '_bin_ovnmtf_news_results.csv'
        params[dataset_name]['k'] = [int(kk)]
        params[dataset_name]['l'] = [int(ll)]
    else:
        filename = dataset_name + '_bin_ovnmtf_news_results.csv'

    out_f = codecs.open(filename, 'w', 'utf-8')
    out_f.write('X,K,L,NMI,RAND,DAVIES\n')
    for k in params[dataset_name]['k']:
        for l in params[dataset_name]['l']:
            for data_str in params[dataset_name]['X']:
                data = eval(data_str)
                data = data.toarray().astype(np.float64)

                h5f = h5py.File('data.h5', 'w')
                h5f.create_dataset('X', data=data.T)
                h5f.close()

                error_best = np.inf
                for _ in xrange(10):
                    tick1 = time.time()
                    # U, S, V, labels_pred, _, error = fnmtf(data, k, l)

                    proc = subprocess.Popen(['./algos', 'bin_ovnmtf', str(k), str(l), '10000'],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

                    (out, err) = proc.communicate()
                    print('out: {}'.format(out))

                    U = np.genfromtxt('U.csv', delimiter=',')
                    S = np.genfromtxt('S.csv', delimiter=',')
                    # V = np.genfromtxt('V.csv', delimiter=',')
                    with open('error.csv') as f:
                        error = float(f.read())
                    labels_pred = np.argmax(U, axis=1)

                    tick2 = time.time()
                    print(u'Took {} secs to train the {} model...'.format((tick2 - tick1), 'bin_ovnmtf'))

                    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
                    rand_score = adjusted_rand_score(labels_true, labels_pred)
                    davies_score = davies_bouldin_score(data, labels_pred, calculate_centroids_doc_mean(data, labels_pred, k))

                    out_f.write(u'{},{},{},{},{},{}\n'.format(data_str, k, l, nmi_score, rand_score, davies_score))

                    print('Execution: X: {}, k: {}, l: {}'.format(data_str, k, l))
                    print('Algo error: {}'.format(error))
                    print('NMI score: {}'.format(nmi_score))
                    print('Rand score: {}'.format(rand_score))
                    print('Davies score: {}'.format(davies_score))
                    print('-----------------------------------------------\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=('ig', 'igtoy', 'newsgroup', 'nips'))
    parser.add_argument('-a', '--algo', choices=('onmtf', 'kmeans', 'fnmtf', 'fkmeans', 'bin_ovnmtf', 'ovnmtf'))
    parser.add_argument('-k', help='number of row clusters', required=False)
    parser.add_argument('-l', help='number of column clusters', required=False)

    args = parser.parse_args()

    # dataset_name = sys.argv[1]
    # algorithm_to_run = sys.argv[2]
    # k = sys.argv[3]

    # print('Could not find algorithm to run argument and/or dataset name!!!')
    # raise SystemError(1)

    current_module = sys.modules[__name__]
    function_to_run_str = 'run_{}'.format(args.algo)
    function_to_run = getattr(current_module, function_to_run_str)

    dataset = get_dataset(args.dataset)
    X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf = preprocess(dataset)
    function_to_run(X_train, X_train_norm, X_train_tfidf, X_train_norm_tfidf, dataset.target, args.dataset, args.k, args.l)

if __name__ == '__main__':
    main()
