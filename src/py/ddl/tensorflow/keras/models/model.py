from ddl.tensorflow.keras.models.model_prebuilder import ModelPreBuilder
from ddl.tensorflow.keras.parallelism.pipeline.layer import PipelineLayer, \
    PipelineInputLayer
from tensorflow.keras.models import Sequential as KerasSequential, Model as \
    KerasModel
from tensorflow.keras.layers import Input


class Model(ModelPreBuilder):
    def __init__(
            self,
            graph_def: callable,
            input_shape: tuple = None
    ):
        """
        keras的模型预构建
        @param graph_def: 模型的层级结构定义函数,
         接收一个参数, 这个参数为inputs张量或者数个inputs张量元组,
         返回一个输出张量或者多个输出张量元组
        @param input_shape: 输入形状, 当模型时第一阶段模型时需要, 将会创建一个Input张量
        """
        assert callable(graph_def)

        if input_shape is not None:
            if isinstance(input_shape[0], (tuple or list)):
                self.__inputs = []
                for each in input_shape:
                    self.__inputs.append(Input(shape=each))
                self.__inputs = tuple(self.__inputs)
            else:
                self.__inputs = (Input(shape=input_shape),)
        else:
            self.__inputs = None
        self.__pipelined_inputs = self.__inputs

        def model_def():
            assert self.__pipelined_inputs is not None

            if isinstance(self.__pipelined_inputs, (list, tuple)):
                outputs = graph_def(*self.__pipelined_inputs)
            else:
                outputs = graph_def(self.__pipelined_inputs)
            return KerasModel(
                inputs=self.__inputs, outputs=outputs
            )

        super().__init__(model_def)

    def set_inputs_shape(self, inputs_shape: tuple):
        """
        留给自动获取上一阶段输出形状并赋值给这一阶段的接口
        @param inputs_shape:
        @return:
        """
        assert self.__inputs is None
        first_element = inputs_shape[0]

        if isinstance(first_element, (tuple, list)):
            self.__pipelined_inputs = []
            pipeline_layers = []
            self.__inputs = []
            for i in range(len(inputs_shape)):
                this_inputs = Input(shape=inputs_shape[i])
                pipeline_layer = PipelineInputLayer(index=i)
                self.__inputs.append(this_inputs)
                pipeline_layers.append(pipeline_layer)
                self.__pipelined_inputs.append(pipeline_layer(this_inputs))
            self.__inputs = tuple(self.__inputs)
            self.__pipelined_inputs = tuple(self.__pipelined_inputs)
            self._set_pipeline_layers(tuple(pipeline_layers))
        else:
            self.__inputs = Input(shape=inputs_shape)
            pipeline_layer = PipelineInputLayer()
            self.__pipelined_inputs = pipeline_layer(self.__inputs)
            self._set_pipeline_layers((pipeline_layer,))


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
        self._set_pipeline_layers((layer,))
        self.__layers = [layer, *self.__layers]
