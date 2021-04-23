class PipelinePipe:
    """
    PipelineStage与PipelineStage之间进行通信的对象
    """

    def __init__(self, shape: tuple, comes_from=None, output_index: int = -1):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        if comes_from is not None:
            assert isinstance(comes_from, PipelineStage)
            self.__index_of = {id(comes_from): output_index}
        else:
            self.__index_of = {}
        self.__comes_from = comes_from
        self.__send_to = []
        self.__shape = shape

    @property
    def shape(self) -> tuple:
        return self.__shape

    @property
    def comes_from(self):
        return self.__comes_from

    @property
    def send_to(self) -> tuple:
        return tuple(self.__send_to)

    def index_of(self, stage):
        return self.__index_of[id(stage)]

    def send_to_stage(self, stage, input_index: int):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        assert id(stage) not in self.__index_of.keys(), \
            'a PipelinePipe can not connect a PipelineStage more than once'
        self.__index_of[id(stage)] = input_index
        self.__send_to.append(stage)

    def __str__(self):
        comes_from_str = id(self.comes_from) \
            if self.comes_from is not None else None
        send_to_str = ','.join([str(id(each)) for each in self.__send_to])
        return \
            f'PipelinePipe{{' \
            f' comes_from: {comes_from_str},' \
            f' shape: {self.__shape}, send_to: {send_to_str}' \
            f' }}'

    def __repr__(self):
        return self.__str__()


class PipelineInput(PipelinePipe):
    def __init__(self, shape: tuple):
        super().__init__(shape)
