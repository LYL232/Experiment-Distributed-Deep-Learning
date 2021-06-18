from tensorflow.python.eager import context
import tensorflow as tf
from typing import Tuple


def executing_eagerly():
    return context.executing_eagerly()


def make_tf_function(f):
    return tf.function(f)


def formalize_shapes(shapes) -> Tuple[Tuple[int]]:
    """
    一般化形状的形式: 将输入转变为数个形状整合到一起的输出
    如果是一般的元素, 必须能转换成整数, 那么返回这个元素的元组的元组: 32 => ((32,),)
    如果是元组或者列表, 那么返回长度相当的元组, 该元组的每个元素仍是元组
        ((12, 12), (13,), 14) => ((12, 12), (13,), (14,))
        (12, 13, 12) => ((12, 13, 12),)
    @param shapes:
    @return: 形式化后的shape元组的元组,
    """
    res = []
    if isinstance(shapes, (tuple, list)):
        assert len(shapes) > 0
        if not isinstance(shapes[0], (tuple, list)):
            shapes = (shapes,)
        for each_shape in shapes:
            element = []
            for each in each_shape:
                element.append(int(each))
            res.append(tuple(element))
    else:
        res.append((int(shapes),))
    return tuple(res)
