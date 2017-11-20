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

__function__= ["ROSENBROCK FUNCTION"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util


ndim = 100


def rosenbrock(xx):

    d = len(xx)
    sum_ = 0
    for ii in range(d - 1):
        xi = xx[ii] * 7.5 + 2.5
        xiplus1 = xx[ii + 1] * 7.5 + 2.5
        new = 100.0 * (xiplus1 - xi ** 2) ** 2 + (xi - 1) ** 2
        sum_ += new

    normalizer = 50000.0 / ((90 ** 2 + 9 ** 2) * (ndim - 1))
    return sum_ * normalizer


def main(params, **kwargs):
    print 'Params: ', params
    print 'kwargs: ', kwargs

    xx = []
    for i in range(1, ndim + 1):
        xx.append(float(params["x" + str(i)]))

    y = rosenbrock(xx)
    return y


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
