'''
(1)算子信息库json命名格式：
   算子名称
		attrs
			attrs0
				type     str         bool	  float 或者 int         list or tuple                     axis
				value [字符串列表]    无  [最小值,最大值]   [列表的最大长度,最小值,最大值]  无(0维时候为None，别的随机(-r,r-1))
			attrs1
			attrs2
		inputs
			input0
				shape: Tensor:[最小维度,最大维度](标量Tensor取为[0,0])     输入是axis(int,list,tuple): 'axis_list_tuple'
				dtype 支持的类型
                dynamic_inputs 如果算子是变输入,标注,为最大的输入个数
			input1
		    ...
(2)所有的算子网络部分都需要函数入口 attributes=None, inputs=None
(3)泛化框架限制：无法隐式类型转换；无法广播机制；无法输入有约束,对于输入有约束的可以用框架产生手动修改；异常用例需要根据具体算子增加，框架只覆盖dtype的异常用例
'''
import argparse
import json
import random
import numpy as np
import re

parser = argparse.ArgumentParser(description="Create test case")
parser.add_argument("op_name", type=str, help="the operation to test")
parser.add_argument("grad", type=str, help="Choose in ['forward', 'grad']")
# python generate_ops.py ReduceSum grad

args_opt = parser.parse_args()
opname = args_opt.op_name
grad = args_opt.grad

with open(r"./ops_info.json", 'r', encoding='utf8')as fp:
    op_json = json.load(fp)[str(opname)]

var_import = (
    "from ..share.ops.primitive.{}_ops import {}Mock\n"
    "from ..share.ops.primitive.{}_ops import {}\n"
).format(opname.lower(), opname, opname.lower(), opname)

const_import = (
    "from mindspore.common import Tensor\n"
    "from mindspore.common import dtype as mstype\n"
    "import numpy as np\n"
    "import pytest\n\n\n\n"
)


def get_summary(op_name, s_dtype, s_rank):
    return ("'''\nTEST_SUMMARY: test {} with input shape from {}d, dtype {}\n"
            "\nPreCondition:\n"
            "    1. install mindspore whl\n"
            "    2. set env variables to specify the backend and mode\n"
            "Test Steps:\n").format(op_name, s_rank, s_dtype) + \
           ("    1. create a net which contains op {}\n"
            "    2. give the input data: the type of inputs is {}, shape {}d\n"
            "    3. compute mindspore and benchmark forward/grad result,"
            "the compare their results.\n"
            "Expected Result:\n"
            "    1.  output return ok and the accuracy is consistent with the benchmark.\n"
            "'''\n\n\n") \
               .format(op_name, s_dtype, s_rank)


def get_summary_errorcase(op_name, wrong_dtype):
    return ("'''\nTEST_SUMMARY: test {} with input dtype {}\n"
            "\nPreCondition:\n"
            "    1. install mindspore whl\n"
            "    2. set env variables to specify the backend and mode\n"
            "Test Steps:\n").format(op_name, wrong_dtype) + \
           ("    1. create a net which contains op {}\n"
            "    2. give the input data dtype {}\n"
            "Expected Result:\n"
            "    1.  Throws TypeError, RuntimeError or ValueError and gives instructive prompts.\n"
            "'''\n\n\n") \
               .format(op_name, wrong_dtype)


def get_casename(op_name, case_dtype, case_rank):
    return ("def test_{}_input_dtype_{}_{}d():\n"
            "    input_list = []\n"). \
        format(op_name.lower(), case_dtype, case_rank)

#产生各种Tensor输入(可自行根据算子补充)
def get_input(r, d, i):
    a = []
	#产生shape，shape的取值为1到512，value满足(0,1)的正态分布
    shape = tuple([random.randint(1, 512) for _ in range(r)])
    if r == 0:
        if d in ["complex64", "complex128"]:
            x = "    x" + str(i) + "_real = np.random.randn{}\n".format(shape)
            a.append(x)
            x = "    x" + str(i) + "_imag = np.random.randn{}\n".format(shape)
            a.append(x)
            x = "    x" + str(i) + " = Tensor(x{}_real + 1j*x{}_imag, dtype=mstype.{})\n".format(
                i, i, d)
        else:
            x = "    x" + str(i) + " = Tensor(np.random.randn{}, dtype=mstype.{})\n".format(shape,
                                                                                            d)
    else:
        if d in ["complex64", "complex128"]:
            x = "    x" + str(i) + "_real = np.random.randn{}.astype(np.float32)\n".format(shape)
            a.append(x)
            x = "    x" + str(i) + "_imag = np.random.randn{}.astype(np.float32)\n".format(shape)
            a.append(x)
            x = "    x" + str(i) + " = Tensor(x{}_real + 1j*x{}_imag, dtype=mstype.{})\n".format(
                i, i, d)
        else:
            x = "    x" + str(i) + " = Tensor(np.random.randn{}.astype(np.{}))\n".format(shape, d)
    a.append(x)
    x = "    input_list.append(x" + str(i) + ")\n"
    a.append(x)
    return a

#产生axis输入，int或者list或者tuple
def get_input_axis(r, d, i):
    a = []
    if r == 0:
        x = "    x" + str(i) + " = None\n"
    else:
        if d == 'axis_list_tuple':
            a0 = [random.randint((-r), (r - 1)) for _ in range(random.randint(1, r))]
            a1 = tuple([random.randint((-r), (r - 1)) for _ in range(random.randint(1, r))])
            a2 = random.randint((-r), (r - 1))
            a3 = random.choice([a0, a1, a2])
        x = "    x" + str(i) + " = {}\n".format(a3)
    a.append(x)
    x = "    input_list.append(x" + str(i) + ")\n"
    a.append(x)
    return a

#产生各种属性(可自行根据算子补充)
def get_attrs(r):
    if 'attrs' in op_json.keys():
        attributes = {}
        for i in op_json['attrs'].keys():
            if 'value' in op_json['attrs'][str(i)].keys():
                c = op_json['attrs'][str(i)]['value']
            if op_json['attrs'][str(i)]['type'] == 'bool':
                attributes[str(i)] = random.choice([True, False])
            if op_json['attrs'][str(i)]['type'] == 'float':
                attributes[str(i)] = np.random.rand() * (c[1] - c[0]) + c[0]
            if op_json['attrs'][str(i)]['type'] == 'int':
                attributes[str(i)] = random.randint(c[0], c[1])
            if op_json['attrs'][str(i)]['type'] == 'str':
                attributes[str(i)] = random.choice(c)
            if op_json['attrs'][str(i)]['type'] == 'axis':
                if r == 0:
                    attributes[str(i)] = None
                else:
                    attributes[str(i)] = random.randint((-r), (r - 1))
            if op_json['attrs'][str(i)]['type'] == 'list or tuple':
                a0 = [random.randint(c[1], c[2]) for _ in range(random.randint(1, c[0]))]
                a = tuple([random.randint(c[1], c[2]) for _ in range(random.randint(1, c[0]))])
                attributes[str(i)] = random.choice([a0, a])
        a = "    attributes= {}\n".format(attributes)
    else:
        a = "    attributes= None\n"
    return a


def get_mock(op_name):
    return ("    fact = {}Mock(\n"
            "        attributes=attributes,\n"
            "        inputs=input_list)\n"
            "    fact.forward_cmp()\n"
            "    fact.grad_cmp()\n\n\n").format(op_name)


def get_forward(op_name):
    return ("    fact = {}Mock(\n"
            "        attributes=attributes,\n"
            "        inputs=input_list)\n"
            "    fact.forward_cmp()\n\n\n").format(op_name)


def get_errorcase(op_name):
    return ("    with pytest.raises((RuntimeError, TypeError, ValueError)):\n"
            "        fact = {}Mock(\n"
            "            attributes=attributes,\n"
            "            inputs=input_list)\n"
            "        fact.forward_mindspore_impl()\n\n\n").format(op_name)


lines = [var_import + const_import]
dtype_range = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float16",
               "float32", "float64", "bool", "complex64", "complex128"]  # Complex64 Complex128
support_dtypes = list(op_json['inputs']['input0']['dtype'])
unsupport_dtypes = []
for i in dtype_range:
    if i not in support_dtypes:
        unsupport_dtypes.append(i)
case_number = 0

#正常用例
print("Normal cases")
rank_input = []
for i in range(len(op_json['inputs'].keys())):
    rank_input.append(op_json['inputs']['input' + str(i)]['shape'])
for rank in range(rank_input[0][0], (rank_input[0][1] + 1)):
    for dtype in support_dtypes:
        lines.append(get_summary(opname, dtype, rank))
        lines.append(get_casename(opname.lower(), dtype, rank))
        a = get_input(rank, dtype, 0)
        lines.append(a)
		#对于addn,concat这些变输入的算子，随机产生输入
        if "dynamic_inputs" in op_json['inputs']['input0'].keys():
            num = random.randint(1, op_json['inputs']['input0']['dynamic_inputs'])
            for i in range(1, num):
                for n in range((4 if dtype in ["complex64", "complex128"] else 2)):
                    c = re.sub('x0', ('x0' + str(i)), a[n])
                    lines.append(c)
        for i in range(1, len(rank_input)):
			#输入为同shape同dtype
            if rank_input[i][1] == rank_input[0][1] and rank_input[i][0] == rank_input[0][0]:
                for n in range((4 if dtype in ["complex64", "complex128"] else 2)):
                    c = re.sub('x0', ('x' + str(i)), a[n])
                    lines.append(c)
			#输入为axis
            elif rank_input[i] == 'axis_list_tuple':
                lines.append(get_input_axis(rank, rank_input[i], i))
			#输入为同type不同shape
            else:
                ranks = random.choice(np.arange(rank_input[i][0], rank_input[i][1] + 1))
                lines.append(get_input(ranks, dtype, i))
        lines.append(get_attrs(rank))
		#算子正反向
        if grad == 'grad':
            lines.append(get_mock(opname))
        else:
            lines.append(get_forward(opname))
        case_number += 1

#异常用例(只考虑dtype，具体算子需要具体补充异常场景)
print("Abnormal cases for dtype")
for dtype in unsupport_dtypes:
    lines.append(get_summary_errorcase(opname, dtype))
    lines.append(get_casename(opname.lower(), dtype, 3))
    a = get_input(3, dtype, 0)
    lines.append(a)
    for i in range(1, len(rank_input)):
        if rank_input[i][1] == rank_input[0][1] and rank_input[i][0] == rank_input[0][0]:
            for n in range((4 if dtype in ["complex64", "complex128"] else 2)):
                c = re.sub('x0', ('x' + str(i)), a[n])
                lines.append(c)
        elif rank_input[i] == 'axis_list_tuple':
            lines.append(get_input_axis(rank, rank_input[i], i))
        else:
            ranks = random.choice(np.arange(rank_input[i][0], rank_input[i][1] + 1))
            lines.append(get_input(ranks, dtype, i))
    lines.append(get_attrs(rank))
    lines.append(get_errorcase(opname))
    case_number += 1

file_name = "test_{}.py".format(opname.lower())

fout = open(file_name, "w")
for i in range(len(lines)):
    fout.writelines(lines[i])
fout.close()

print("case_number: ", case_number)
