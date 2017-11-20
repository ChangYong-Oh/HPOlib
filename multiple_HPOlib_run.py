import time
import sys
import os.path
import subprocess


def multiple_runs(optimizer_dir, working_dir):
	optimizer_dir = os.path.realpath(optimizer_dir)
	working_dir = os.path.realpath(working_dir)
	n_sample_run = 5
	process_list = []
	for i in range(n_sample_run):
		cmd_str = 'HPOlib-run -o ' + optimizer_dir + ' --cwd ' + working_dir
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


if __name__ == '__main__':
	multiple_runs(sys.argv[1], sys.argv[2])
