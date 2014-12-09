
import numpy as np
import random

from matplotlib import pyplot as plt

class Bicluster(object):

    def __init__(self, A, I, J):
        self.A = A
        self.I = I
        self.J = J



class MSR(object):

    def __init__(self, data):
        self.data = data
        # print "calculating aiJ, aIj and aIJ..."
        self.n, self.m = data.shape
        self.aiJ = np.mean(data, axis=1)
        self.aIj = np.mean(data, axis=0)
        self.aIJ = np.mean(data)
        self._H = None
        self._HiJ = None
        self._HIj = None


    @property
    def H(self):
        if self._H is None:
            print "computing MSR..."
            self._H = self._compute_H()
            print "MSR value: %s" % self._H
        return self._H


    @property
    def HiJ(self):
        if self._HiJ is None:
            self._HiJ = self._compute_HiJ()
        return self._HiJ


    @property 
    def HIj(self):
        if self._HIj is None:
            self._HIj = self._compute_HIj()
        return self._HIj


    def _compute_H(self):
        # it could be changed to change the shape of aiJ and aIj to simplify the code
        #   by calculating H in a vectorized way, but it'd spend more memory
        H = 0
        for i in xrange(self.n):
            for j in xrange(self.m):
                H += ( self.data[i,j] - self.aIj[j] - self.aiJ[i] + self.aIJ ) ** 2
        H *= 1.0/(self.n * self.m)
        return H


    def _compute_HiJ(self):
        HiJ = np.zeros(self.n)
        for i in xrange(self.n):
            for j in xrange(self.m):
                HiJ[i] += ( self.data[i,j] - self.aIj[j] - self.aiJ[i] + self.aIJ )**2
        HiJ *= 1.0/self.m
        return HiJ


    def _compute_HIj(self):
        HIj = np.zeros(self.m)
        for j in xrange(self.m):
            for i in xrange(self.n):
                HIj[j] += ( self.data[i,j] - self.aIj[j] - self.aiJ[i] + self.aIJ )**2
        HIj *= 1.0/self.n
        return HIj



class DeltaBiclustering(object):

    def remove_unique_nodes(self, data, delta, I=None, J=None):
        it = 1

        if I is None:
            I = np.arange(len(data))
        if J is None:
            J = np.arange(len(data[0]))

        while True:
            print "%s iteration of unique node deletion..." % it
            it += 1

            msr = MSR(data[I][:,J])

            if msr.H < delta:
                break

            if len(I) == 1:
                break

            if len(J) == 1:
                break

            row_idx_to_remove = np.argmax(msr.HiJ)
            col_idx_to_remove = np.argmax(msr.HIj)
            if msr.HiJ[row_idx_to_remove] > msr.HIj[col_idx_to_remove]:
                print "removing %s row." % row_idx_to_remove
                I = np.delete(I, row_idx_to_remove)
            else:
                print "removing %s column." % row_idx_to_remove
                J = np.delete(J, col_idx_to_remove)

        return (data[I][:,J],I,J)


    def remove_multiple_nodes(self, data, delta, alpha, I=None, J=None):
        it = 1
        
        if I is None:
            I = np.arange(len(data))
        if J is None:
            J = np.arange(len(data[0]))

        while True:
            removal_ocurred = True

            print "%s iteration of multiple node deletion..." % it
            it += 1

            msr = MSR(data[I][:,J])

            if msr.H < delta:
                break

            if len(I) == 1:
                break

            if len(J) == 1:
                break

            # remove lines first
            rows_idxs_to_remove = np.nonzero(msr.HiJ > alpha*msr.H)[0]
            if len(rows_idxs_to_remove) > 0:
                print "removing %s rows..." % len(rows_idxs_to_remove)
                I = np.delete(I, rows_idxs_to_remove)
            else:
                print "rows removal haven't ocurred..."
                removal_ocurred = False

            msr = MSR(data[I][:,J])
            cols_idxs_to_remove = np.nonzero(msr.HIj > alpha*msr.H)[0]
            if len(cols_idxs_to_remove) > 0:
                print "removing %s columns..." % len(cols_idxs_to_remove)
                J = np.delete(J, cols_idxs_to_remove)
            else:
                print "columns removal haven't ocurred..."
                if not removal_ocurred:
                    return self.remove_unique_nodes(data, delta, I, J)

        return (data[I][:,J],I,J)


    def add_nodes(self, data, orig_data, alpha, I, J):
        it = 1

        orig_n, orig_m = orig_data.shape

        orig_I = np.arange(orig_n)
        orig_J = np.arange(orig_m)

        I_not_in_bicluster = np.setdiff1d(orig_I, I)
        J_not_in_bicluster = np.setdiff1d(orig_J, J)

        addition_ocurred = True
        while addition_ocurred:
            print "%s iteration of node addition..." % it
            it += 1

            addition_ocurred = False

            msr = MSR(orig_data[I][:,J])

            # column addition
            for j in J_not_in_bicluster:
                data_with_j = np.column_stack((data, orig_data[I][:,j]))
                # msr_with_j = MSR(data_with_j)
                new_col = orig_data[I][:,j]
                msr_with_j = np.mean( (new_col - msr.aiJ - np.mean(new_col) + msr.aIJ ) ** 2 )
                if msr_with_j <= msr.H:
                    J = np.append(J, j)
                    J_not_in_bicluster = np.delete(J_not_in_bicluster, np.where(j == J_not_in_bicluster)[0])
                    data = data_with_j
                    addition_ocurred = True

            msr = MSR(orig_data[I][:,J])

            # row addition
            for i in I_not_in_bicluster:
                data_with_i = np.row_stack((data, orig_data[i,J]))
                # msr_with_i = MSR(data_with_i)
                new_row = orig_data[i,J]
                msr_with_i = np.mean( (new_row - np.mean(new_row) - msr.aIj + msr.aIJ ) ** 2 )
                if msr_with_i <= msr.H:
                    I = np.append(I, i)
                    I_not_in_bicluster = np.delete(I_not_in_bicluster, np.where(i == I_not_in_bicluster)[0])
                    data = data_with_i
                    adition_ocurred = True

    # My guess is that we don't need invertions, because we are not in genes expression domain        
            # inverted row addition
    #         msr = MSR(orig_data)
    #         rows_idxs_to_add_overlap = np.nonzero(msr.HIj <= alpha*msr.H)[0]
    #         rows_idxs_to_add = np.setdiff1d(rows_idxs_to_add_overlap, new_I)
    #         if len(rows_idxs_to_add) > 0:
    #             new_I = np.append(new_I, rows_idxs_to_add)
    #             inverted_rows = np.fliplr(orig_data[rows_idxs_to_add][:,J])
    #             data = np.row_stack((data, inverted_rows))
    #         else:
    #             addition_ocurred = False

        return (orig_data[I][:,J],I,J)


    def mask(self, data, I, J, min_v, max_v):
        for i in I:
            for j in J:
                data[i,j] = random.uniform(min_v, max_v)
        return data


    def find_biclusters(self, A, delta, alpha, num_biclusters):
        biclusters = []

        min_v = np.min(A)
        max_v = np.max(A)

        Aline = A.copy()

        for i in range(num_biclusters):
            n, m = Aline.shape
            print Aline.shape

            plt.matshow(Aline, cmap=plt.cm.Blues, vmin=min_v, vmax=max_v)
            plt.show()

            print "--- Looking for bicluster %s:" % (i+1)

            # do not perform multiple node deletion
            if len(A) < 100 or len(A[0]) < 100:
                (C,I,J) = self.remove_unique_nodes(Aline, delta)
            else:
                (C,I,J) = self.remove_multiple_nodes(Aline, delta, alpha)

            D = self.add_nodes(C, A, alpha, I, J)

            biclusters.append(C)

            plt.matshow(C, cmap=plt.cm.Blues, vmin=min_v, vmax=max_v)
            plt.show()

            Aline = self.mask(Aline, I, J, min_v, max_v)

        plt.matshow(Aline, cmap=plt.cm.Blues, vmin=min_v, vmax=max_v)
        plt.show()

        return biclusters
