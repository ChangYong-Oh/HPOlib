

def generate_pb(ndim):
	pb_file = open('config.pb', 'w')

	text = "language: PYTHON\nname:     \"spearmint_to_HPOlib\"\n\n"

	for d in range(1, ndim + 1):
		text += "variable {\n"
		text += " name: \"x" + str(d) + "\"\n"
		text += " type: FLOAT\n"
		text += " size: 1\n"
		text += " min:  -1\n"
		text += " max:  1\n"
		text += "}\n\n"

	pb_file.writelines(text)

	pb_file.close()


if __name__ == '__main__':
	generate_pb(1000)