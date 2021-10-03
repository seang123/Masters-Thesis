
from abc import ABC


class DataLoader(ABC):

    @abstractmethod
    def load_data_train():
        pass

    @abstractmethod
    def load_data_val():
        pass

    @abstractmethod
    def load_data():
        pass

