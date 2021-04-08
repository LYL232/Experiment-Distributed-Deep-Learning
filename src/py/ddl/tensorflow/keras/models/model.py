from ddl.tensorflow.keras.models.model_prebuilder import ModelPreBuilder
from ddl.tensorflow.keras.parallelism.pipeline.layer import PipelineLayer
from tensorflow.keras.models import Model as KerasModel, \
    Sequential as KerasSequential


class Model(ModelPreBuilder):
    def __init__(self, **kwargs):
        # 在流水线模型中非第一阶段的inputs需要将其梯度传递给上一阶段，所以需要替换其inputs
        self.__inputs = kwargs.pop('inputs', None)

        super().__init__(
            lambda: KerasModel(inputs=self.__inputs, **kwargs)
        )


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
