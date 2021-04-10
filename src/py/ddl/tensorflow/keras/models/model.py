from ddl.tensorflow.keras.models.model_prebuilder import ModelPreBuilder
from ddl.tensorflow.keras.parallelism.pipeline.layer import PipelineLayer, \
    PipelineInputLayer
from tensorflow.keras.models import Sequential as KerasSequential, Model as \
    KerasModel
from tensorflow.keras.layers import Input
from tensorflow.python.framework.ops import Tensor


class Model(ModelPreBuilder):
    def __init__(
            self,
            graph_def: callable,
            inputs: Tensor or tuple = None
    ):
        """
        keras的模型预构建
        @param inputs: 模型的原始输入
        @param graph_def: 模型的层级结构定义函数,
         接收一个参数, 这个参数为inputs张量或者数个inputs张量元组,
         返回一个输出张量或者多个输出张量元组
        """
        assert callable(graph_def)

        if inputs is not None:
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                self.__inputs = []
                for i in range(len(inputs)):
                    self.__inputs.append(PipelineInputLayer(index=i)(inputs[i]))
                    self.__inputs = tuple(self.__inputs)
            else:
                self.__inputs = PipelineInputLayer()(inputs)

        else:
            self.__inputs = None

        def model_def():
            assert self.__inputs is not None
            return KerasModel(
                inputs=self.__inputs, outputs=graph_def(self.__inputs))

        super().__init__(model_def)

    def set_inputs_shape(self, inputs_shape: tuple):
        """
        留给自动获取上一阶段输出形状并赋值给这一阶段的接口
        @param inputs_shape:
        @return:
        """
        assert self.__inputs is None
        if isinstance(inputs_shape, tuple) or isinstance(inputs_shape, list):
            self.__inputs = []
            for i in range(len(inputs_shape)):
                self.__inputs.append(
                    PipelineInputLayer(index=i)(Input(shape=inputs_shape[i]))
                )
            self.__inputs = tuple(self.__inputs)
        else:
            self.__inputs = PipelineInputLayer()(Input(shape=inputs_shape))


class Sequential(ModelPreBuilder):
    def __init__(self, layers=None, name=None):
        self.__layers = layers

        super().__init__(
            lambda: KerasSequential(layers=self.__layers, name=name)
        )

    def insert_pipeline_input_layer(self, layer: PipelineLayer):
        """
        插入一个流水线输入层, 用于阶段与阶段之间交流, 要求调用时built必须为False, 因为模型一旦
        构建, 再修改构建时的参数就无意义了
        @param layer: 流水线交流层
        @return: None
        """
        assert not self.built
        self.__layers = [layer, *self.__layers]
