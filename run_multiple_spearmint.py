import time
from datetime import datetime
import sys
import os.path
import subprocess
import numpy as np


def run_multiple_spearmint_init(optimizer_dir, working_dir):
	optimizer_dir = os.path.realpath(optimizer_dir)
	working_dir = os.path.realpath(working_dir)
	n_sample_run = 5
	process_list = []
	for i in range(n_sample_run):
		cmd_str = 'HPOlib-run -o ' + optimizer_dir + ' --cwd ' + working_dir + ' -s ' + str(np.random.randint(0, 20000))
		f = open(os.devnull, 'w')
		process = subprocess.Popen(cmd_str, shell=True, stdout=f, stderr=f)
		process_list.append(process)
	n_running = n_sample_run
	while n_running > 0:
		time.sleep(60)
		running = [elm.poll() is None for elm in process_list]
		n_running = running.count(True)
		print('%d/%d is still running %s' % (n_running, n_sample_run, time.strftime("%H:%M:%S")))
		sys.stdout.flush()
	print('Done')


def run_multiple_spearmint_continue(optimizer_dir, exp_dir_list):
	process_list = []
	n_exp = len(exp_dir_list)
	for exp_dir in exp_dir_list:
		cmd_str = 'HPOlib-run -o ' + os.path.realpath(optimizer_dir) + ' -r ' + os.path.realpath(exp_dir)
		f = open(os.devnull, 'w')
		process = subprocess.Popen(cmd_str, shell=True, stdout=f, stderr=f)
		process_list.append(process)

	n_running = n_exp
	while n_running > 0:
		time.sleep(60)
		n_running = [elm.poll() is None for elm in process_list].count(True)
		print('Currently running experiments(%s)[%d/%d]' % (datetime.now().strftime('%Y%m%d-%H:%M:%S:%f'), n_running, n_exp))
		for i in range(n_exp):
			if process_list[i].poll() is None:
				print('    %s' % exp_dir_list[i].name)
		sys.stdout.flush()


if __name__ == '__main__':
	if len(sys.argv) == 3:
		run_multiple_spearmint_init(sys.argv[1], sys.argv[2])
	else:
		run_multiple_spearmint_continue(sys.argv[1], sys.argv[2:])
