##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__function__= ["ROTATEDSTYBLINSKI-TANG FUNCTION"]

import time
import numpy as np

import HPOlib.benchmarks.benchmark_util as benchmark_util


ndim = 1000


def generate_orthogonal_matrix(ndim):
    x = np.exp(np.sin(np.linspace(-ndim ** 0.5, ndim ** 0.5, ndim)))
    x_repeated = np.tile(x, (ndim, 1))
    gram_mat = np.exp(-(x_repeated - x_repeated.T) ** 2)
    return np.linalg.qr(gram_mat, 'complete')[0]

orthogonal_matrix = generate_orthogonal_matrix(ndim)


def rotatedstyblinskitang(xx):

    d = len(xx)
    sum_ = 0
    xx = orthogonal_matrix.dot(xx)
    for ii in range(d):
        xi = xx[ii]
        new = 312.5 * xi ** 4 - 200.0 * xi ** 2 + 12.5 * xi
        sum_ += new

    return sum_ / float(d)


def main(params, **kwargs):
    print 'Params: ', params
    print 'kwargs: ', kwargs

    xx = []
    for i in range(1, ndim + 1):
        xx.append(float(params["x" + str(i)]))

    y = rotatedstyblinskitang(xx)
    return y


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
