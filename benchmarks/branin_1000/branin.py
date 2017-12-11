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

__function__= ["BRANIN FUNCTION"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util


ndim = 1000



def branin(xx):
    import math

    a = 1
    b = 5.1 / (4 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6
    s = 10
    t = 1.0 / (8 * math.pi)

    d = len(xx)
    n_repeat = d / 2
    sum_ = 0
    for ii in range(n_repeat):
        xi0 = xx[2 * ii] * 7.5 + 2.5
        xi1 = xx[2 * ii + 1] * 7.5 + 7.5
        new = a * (xi1 - b * xi0 ** 2 + c * xi0 - r) ** 2 + s * (1 - t) * math.cos(xi0) + s
        sum_ += new
    sum_ /= float(n_repeat)

    return sum_


def main(params, **kwargs):
    print 'Params: ', params
    print 'kwargs: ', kwargs

    xx = []
    for i in range(1, ndim + 1):
        xx.append(float(params["x" + str(i)]))

    y = branin(xx)
    return y


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
