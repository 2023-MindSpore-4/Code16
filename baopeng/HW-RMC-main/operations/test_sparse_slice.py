from mindspore import Tensor
from mindspore.common import dtype as mstype
import numpy as np
import pytest
from ..share import AnyNetFactory
from ..share.ops.primitive.sparse_slice_ops import SparseSlice
from ..share.ops.primitive.sparse_slice_ops import SparseSliceMock

'''
TEST_SUMMARY: test SparseSlice with the input tensor are all right
'''


def test_sparse_slice_512_int8():
    input_indices = Tensor(np.random.randint(1, 1024, size=(1024, 2)).astype(np.int64))
    input_values = Tensor(np.random.randn(1024).astype(np.int8))
    input_shape = Tensor(np.random.randint(1020, 1024, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(800, 900, size=(2)).astype(np.int64))
    
    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_uint8():
    input_indices = Tensor(np.random.randint(1, 2048, size=(2048, 3)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 1024, size=(2048)).astype(np.uint8))
    input_shape = Tensor(np.random.randint(2020, 2048, size=(3)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(3)).astype(np.int64))
    input_size = Tensor(np.random.randint(1200, 1500, size=(3)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_512_int16():
    input_indices = Tensor(np.random.randint(1, 1024, size=(512, 4)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 1024, size=(512)).astype(np.int16))
    input_shape = Tensor(np.random.randint(512, 1024, size=(4)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(4)).astype(np.int64))
    input_size = Tensor(np.random.randint(500, 717, size=(4)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_1024_uint16():
    input_indices = Tensor(np.random.randint(1, 1024, size=(1024, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 1024, size=(1024)).astype(np.uint16))
    input_shape = Tensor(np.random.randint(1024, 1280, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(500, 717, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()
    

def test_sparse_slice_2048_int32():
    input_indices = Tensor(np.random.randint(1, 2048, size=(2048, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.int32))
    input_shape = Tensor(np.random.randint(2048, 3072, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1000, 1500, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_4096_int64():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int32))
    input_shape = Tensor(np.random.randint(4096, 5120, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1000, 1500, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_512_float16():
    input_indices = Tensor(np.random.randint(1, 1024, size=(512, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 1024, size=(512)).astype(np.float16))
    input_shape = Tensor(np.random.randint(512, 1024, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(200, 500, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_1024_float():
    input_indices = Tensor(np.random.randint(1, 1024, size=(1024, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 1024, size=(1024)).astype(np.float32))
    input_shape = Tensor(np.random.randint(1024, 2048, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(500, 717, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()
    

def test_sparse_slice_2048_double():
    input_indices = Tensor(np.random.randint(1, 2048, size=(2048, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.float64))
    input_shape = Tensor(np.random.randint(2048, 3072, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1000, 1800, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_4096_complex64():
    x_real = np.random.randn(4096).astype(np.float32)
    x_img = np.random.randn(4096).astype(np.float32)
    input_values = Tensor(x_img + 1j* x_img, dtype=mstype.complex64)
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 5120, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1000, 1800, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_4096_complex128():
    x_real = np.random.randint(1, 2048, size=(4096)).astype(np.float64)
    x_img = np.random.randint(1, 2048, size=(4096)).astype(np.float64)
    input_values = Tensor(x_img + 1j* x_img, dtype=mstype.complex128)
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 5120, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1000, 1800, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()
    fact.grad_cmp()


def test_sparse_slice_1024_bool():
    input_indices = Tensor(np.random.randint(1, 1024, size=(1024, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 1024, size=(1024)).astype(np.bool))
    input_shape = Tensor(np.random.randint(2048, 3072, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 100, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(500, 800, size=(2)).astype(np.int64))

    fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
    fact.forward_cmp()



'''
TEST_SUMMARY: test SparseSlice with indices out of bounds
异常样例
'''


def test_sparse_slice_512_input_indices_out_of_bounds():
    input_indices = Tensor(np.random.randint(2048, 8192, size=(2048, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.float64))
    input_shape = Tensor(np.random.randint(2048, 4096, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))

    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with the indices shape 1D tensor
异常样例
'''


def test_sparse_slice_input_indices_tensor_1D():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with the values shape 2D tensor
异常样例
'''


def test_sparse_slice_input_values_tensor_2D():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096,2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096,2)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with the input shape 2D tensor
异常样例
'''


def test_sparse_slice_input_shape_tensor_2D():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096,2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2, 2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with the start shape 2D tensor
异常样例
'''


def test_sparse_slice_input_start_tensor_2D():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096,2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(1, 2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with the size shape 2D tensor
异常样例
'''


def test_sparse_slice_input_size_tensor_2D():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096,2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(1, 2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with different dtype(indices)
异常样例
'''


def test_sparse_slice_input_indices_dtype_diff():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.float16))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(TypeError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with different dtype(shape)
异常样例
'''


def test_sparse_slice_input_shape_dtype_diff():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.uint8))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(TypeError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with different dtype(start)
异常样例
'''


def test_sparse_slice_input_start_dtype_diff():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int16))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(TypeError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with different dtype(size)
异常样例
'''


def test_sparse_slice_input_size_dtype_diff():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(4096)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.double))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(TypeError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: the number of indices not equal to the number of values
异常样例
'''


def test_sparse_slice_input_indices_notequal_values():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: the shape of indices[1] not equal to shape[0]
异常样例
'''


def test_sparse_slice_input_indices_shape_notequal_shape():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 3)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: shape not equal to start
异常样例
'''


def test_sparse_slice_input_shape_notequal_start():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(3)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: shape not equal to size
异常样例
'''


def test_sparse_slice_input_shape_notequal_size():
    input_indices = Tensor(np.random.randint(1, 2048, size=(4096, 2)).astype(np.int64))
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(3)).astype(np.int64))
    with pytest.raises(ValueError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()


'''
TEST_SUMMARY: test SparseSlice with list
异常样例
'''


def test_sparse_slice_input_list():
    input_indices = [np.random.randint(1, 2048, size=(4096, 2))]
    input_values = Tensor(np.random.randint(1, 2048, size=(2048)).astype(np.int64))
    input_shape = Tensor(np.random.randint(4096, 8192, size=(2)).astype(np.int64))
    input_start = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    input_size = Tensor(np.random.randint(1, 1024, size=(2)).astype(np.int64))
    with pytest.raises(AttributeError):
        fact = SparseSliceMock(inputs=[input_indices, input_values, input_shape, input_start, input_size])
        fact.forward_cmp()
        fact.grad_cmp()
