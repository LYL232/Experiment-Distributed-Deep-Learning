"""
分布式环境并实现分布式流水线并行训练tensorflow2模型

通信方式: mpi

实现思路: 将keras_pipeline_baseline.py 中定义的模型拆分到两个独立的进程上运行,
    这两个模型分别称为FirstStage(包含原始模型的输入)和SecondStage(包含原始模型的输出)
    这两个模型使用同一种优化器,
    当前向传播时, SecondStage请求FirstStage按照batch_size取出数据,
    完成FirstStage的前向传播计算, 将其输出作为输入, 完成前向传播,
    当后向传播时, SecondStage的优化器即将应用(apply)梯度时, 也即模型的权重即将改变时, 将
    计算出的偏差(其实不止偏差)传输到FirstStage, FirstStage获得偏差, 使用偏差计算其loss值,
    再调用其优化器对FirstStage模型进行优化
"""

batch_size = 200

# 总共的样例个数, 写到这里是因为SecondStage模型并不需要载入数据
samples = 60000

data_shape = (28, 28)

lr = 0.001

epochs = 24

# Pipe发送的消息类型枚举定义
# SecondStage请求FirstStage取出数据计算前向传播, 并等待其传输结果
MSG_GET_FORWARD_PROPAGATION = 0
# FirstStage返回其前项传播的结果给SecondStage
MSG_FORWARD_PROPAGATION_RESULT = 1
# SecondStage将后向传播给FirstStage的偏差数据
MSG_GRADIENTS_BACK_PROPAGATION = 2
# 完成训练
MSG_DONE = 3

# 是否打印log到日志输出中
verbose = True


def main():
    from ddl.examples.keras_pipeline_mpi.rank0 import main as main_0
    from ddl.examples.keras_pipeline_mpi.rank1 import main as main_1
    from ddl.tensorflow.communicator import Communicator
    world = Communicator.world()
    if world.rank == 0:
        main_0()
    elif world.rank == 1:
        main_1()
    else:
        raise Exception(f'unexpected rank: {world.rank}')


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(
        os.path.abspath(
            os.path.join(
                __file__,
                '../../../../'
            )
        )
    )

    main()
