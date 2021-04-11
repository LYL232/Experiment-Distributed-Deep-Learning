from tensorflow.keras.models import Model
import abc


class ModelPreBuilder(metaclass=abc.ABCMeta):
    """
    由于keras Model在定义时就会开始占用资源，比如内存，所以分布式定义时将keras Model的定义
    延后至需要使用时，这个类保存模型用户模型定义时的参数，在使用时再创建Model
    """

    def __init__(self, model_getter: callable):
        assert callable(model_getter)
        self.__model = None
        self.__built = False
        self.__model_getter = model_getter
        self.__pipeline_layers = None

    @property
    def model(self) -> Model:
        if self.__model is None:
            self.__built = True
            self.__model = self.__model_getter()
        return self.__model

    @property
    def built(self) -> bool:
        """
        是否已经构建了模型
        @return: bool
        """
        return self.__built

    @property
    def pipeline_layers(self) -> tuple:
        """
        返回流水线层的元组
        @return: tuple, 如果只有一个, 则是只有一个元素的tuple
        """
        return self.__pipeline_layers

    def _set_pipeline_layers(self, pipeline_layers: tuple):
        """
        由子类在编译时获取流水线层调用
        @param pipeline_layers: tuple, 每个元素必须是PipelineLayer
        @return:
        """
        from ddl.tensorflow.keras.parallelism.pipeline.layer import \
            PipelineLayer
        for each in pipeline_layers:
            assert isinstance(each, PipelineLayer)
        self.__pipeline_layers = pipeline_layers
