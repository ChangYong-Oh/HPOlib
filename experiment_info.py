import os
import numpy as np


def how_many_evaluations():
	benchmarks_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'benchmarks')
	empty_prev = True
	for benchmark in sorted(os.listdir(benchmarks_dir)):
		benchmark_dir = os.path.join(benchmarks_dir, benchmark)
		if os.path.isdir(benchmark_dir):
			if not empty_prev:
				print('-' * 150)
			empty_prev = True
			for run in os.listdir(benchmark_dir):
				run_dir = os.path.join(benchmark_dir, run)
				output_dir = os.path.join(run_dir, 'output')
				jobs_dir = os.path.join(run_dir, 'jobs')
				if run in ['hyperopt_august2013_mod', 'smac_2_10_00-dev', 'spearmint_april2013_mod']:
					pass
				elif os.path.isdir(run_dir) and os.path.exists(output_dir):
					out_file_list = os.listdir(output_dir)
					job_file_list = os.listdir(jobs_dir)
					if len(out_file_list) > 0:
						n_out = len(out_file_list)
						n_job = len(job_file_list)
						print("%-40s %-70s %3d %3d" % (benchmark, run, n_out, n_job))
						empty_prev = False


if __name__ == '__main__':
	how_many_evaluations()