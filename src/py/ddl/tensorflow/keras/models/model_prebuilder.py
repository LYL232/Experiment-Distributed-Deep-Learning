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
