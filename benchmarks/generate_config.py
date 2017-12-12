

def generate_config(ndim):
	pb_file = open('config.pb', 'w')
	pb_content = "language: PYTHON\nname:     \"spearmint_to_HPOlib\"\n\n"
	for d in range(1, ndim + 1):
		pb_content += "variable {\n"
		pb_content += " name: \"x" + str(d) + "\"\n"
		pb_content += " type: FLOAT\n"
		pb_content += " size: 1\n"
		pb_content += " min:  -1\n"
		pb_content += " max:  1\n"
		pb_content += "}\n\n"
	pb_file.writelines(pb_content)
	pb_file.close()

	pcs_file = open('params.pcs', 'w')
	pcs_content = ""
	for d in range(1, ndim + 1):
		pcs_content += 'x' + str(d) + ' real [-1.0, 1.0] [0.0]\n'
	pcs_file.writelines(pcs_content)
	pcs_file.close()

	py_file = open('space.py', 'w')
	py_content = "from hyperopt import hp\n\nspace = {\n"
	for d in range(1, ndim + 1):
		py_content += "\t'x" + str(d) + "': " + "hp.uniform('x" + str(d) + "', -1, 1),\n"
	py_content = py_content[:-2] + '}'
	py_file.writelines(py_content)
	py_file.close()

if __name__ == '__main__':
	generate_config(500)