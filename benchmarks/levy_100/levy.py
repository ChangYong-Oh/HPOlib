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

__function__= ["LEVY FUNCTION"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util


def levy(xx):
    import math

    d = len(xx)
    xi = (xx[0] * 10.0 - 1.0) / 4.0 + 1.0
    sum_ = math.sin(math.pi * xi) ** 2.0
    for ii in range(d - 1):
        xi = (xx[ii] * 10.0 - 1.0) / 4.0 + 1.0
        new = (xi - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * xi + 1.0) ** 2.0)
        sum_ += new
    xi = (xx[d - 1] * 10.0 - 1.0) / 4.0 + 1.0
    new = (xi - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * xi) ** 2.0)
    sum_ += new

    return sum_


def main(params, **kwargs):
    print 'Params: ', params
    print 'kwargs: ', kwargs

    xx = []
    for i in range(1, len(params) + 1):
        xx.append(float(params["x" + str(i)]))

    y = levy(xx)
    return y


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
