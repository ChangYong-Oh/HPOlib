import time
import sys
import os
import os.path
import subprocess
from datetime import datetime
import numpy as np


def run_multiple_treebased(optimizer_dir, benchmark_dir, func_name_list=[]):
	if len(func_name_list) == 0:
		func_name_list = ['levy', 'michalewicz', 'schwefel', 'styblinskitang', 'rosenbrock', 'rotatedstyblinskitang', 'rotatedschwefel']
	ndim_list = [20, 50, 100, 200, 1000]
	n_sample_run = 5

	cmd_str_list = []
	for ndim in ndim_list:
		for func_name in func_name_list:
			for _ in range(n_sample_run):
				cmd_str = 'HPOlib-run -o ' + os.path.realpath(optimizer_dir) + ' --cwd ' + os.path.join(os.path.realpath(benchmark_dir), func_name + str(ndim)) + ' -s ' + str(np.random.randint(0, 20000))
				cmd_str_list.append(cmd_str)

	n_exp = len(ndim_list) * len(func_name_list) * n_sample_run
	n_started = 0
	process_list = []
	beginning_phase = True
	while n_started < n_exp:
		time.sleep(10)
		if [l <= 28 for l in os.getloadavg()].count(False) == 0:
			cmd_str = cmd_str_list[n_started]
			f = open(os.devnull, 'w')
			process_list.append(subprocess.Popen(cmd_str, shell=True, stdout=f, stderr=f))
			n_started += 1
		if os.getloadavg()[0] > 16:
			beginning_phase = False
		if beginning_phase:
			continue
		n_running = [elm.poll() is None for elm in process_list].count(True)
		print('Currently running experiments(%s) %d/%d(running)' % (datetime.now().strftime('%Y%m%d-%H:%M:%S:%f'), n_running, n_exp))
		sys.stdout.flush()

	print('All experiments have started.')

	while [p.poll() for p in process_list].count(None) > 0:
		time.sleep(10)
		n_running = [elm.poll() is None for elm in process_list].count(True)
		print('Currently running experiments(%s) %d/%d(running)' % (datetime.now().strftime('%Y%m%d-%H:%M:%S:%f'), n_running, n_exp))
		sys.stdout.flush()
	print('Done')


if __name__ == '__main__':
	run_multiple_treebased(sys.argv[1], sys.argv[2], sys.argv[3:])