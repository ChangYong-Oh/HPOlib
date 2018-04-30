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

__function__= ["TWOHOOKS 5x5 FUNCTION"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util


def pixel_5by5_group_3_twohooks(xx):
    import numpy as np

    xx = np.array(xx).reshape((5, 5))

    assert np.in1d(xx, ['0', '1', '2']).all()
    assert (len(xx) if isinstance(xx, list) else xx.size) == 5 * 5
    image = np.array([['1', '1', '1', '1', '0'],
                      ['1', '0', '0', '1', '1'],
                      ['1', '1', '0', '2', '2'],
                      ['2', '2', '0', '0', '2'],
                      ['0', '2', '2', '2', '2']])
    return 1.0 - np.sum(np.reshape(xx, (5, 5)) == image) / 25.0


def main(params, **kwargs):
    print 'Params: ', params
    print 'kwargs: ', kwargs

    xx = []
    for i in range(1, len(params) + 1):
        xx.append(params["x" + str(i)])

    y = pixel_5by5_group_3_twohooks(xx)
    return y


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
