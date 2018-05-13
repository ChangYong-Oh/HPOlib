from hyperopt import hp

space = {
	'x1': hp.choice('x1', ['-1', '-2']),
	'x2': hp.choice('x2', ['-1', '-2']),
	'x3': hp.choice('x3', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x4': hp.choice('x4', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x5': hp.choice('x5', ['-1', '-2', '-3']),
	'x6': hp.choice('x6', ['-1', '-2', '-3']),
	'x7': hp.choice('x7', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x8': hp.choice('x8', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x9': hp.choice('x9', ['-1', '-2', '-3', '-4']),
	'x10': hp.choice('x10', ['-1', '-2', '-3', '-4']),
	'x11': hp.choice('x11', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x12': hp.choice('x12', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x13': hp.choice('x13', ['-1', '-2', '-3', '-4', '-5']),
	'x14': hp.choice('x14', ['-1', '-2', '-3', '-4', '-5']),
	'x15': hp.choice('x15', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x16': hp.choice('x16', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x17': hp.choice('x17', ['-1', '-2', '-3', '-4', '-5', '-6']),
	'x18': hp.choice('x18', ['-1', '-2', '-3', '-4', '-5', '-6']),
	'x19': hp.choice('x19', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x20': hp.choice('x20', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x21': hp.choice('x21', ['-1', '-2']),
	'x22': hp.choice('x22', ['-1', '-2']),
	'x23': hp.choice('x23', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x24': hp.choice('x24', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x25': hp.choice('x25', ['-1', '-2', '-3']),
	'x26': hp.choice('x26', ['-1', '-2', '-3']),
	'x27': hp.choice('x27', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x28': hp.choice('x28', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x29': hp.choice('x29', ['-1', '-2', '-3', '-4']),
	'x30': hp.choice('x30', ['-1', '-2', '-3', '-4']),
	'x31': hp.choice('x31', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x32': hp.choice('x32', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x33': hp.choice('x33', ['-1', '-2', '-3', '-4', '-5']),
	'x34': hp.choice('x34', ['-1', '-2', '-3', '-4', '-5']),
	'x35': hp.choice('x35', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x36': hp.choice('x36', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x37': hp.choice('x37', ['-1', '-2', '-3', '-4', '-5', '-6']),
	'x38': hp.choice('x38', ['-1', '-2', '-3', '-4', '-5', '-6']),
	'x39': hp.choice('x39', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3']),
	'x40': hp.choice('x40', ['sep_conv_3', 'sep_conv_5', 'id', 'avg_pool_3', 'max_pool_3'])}