from abc import ABC
from abc import abstractmethod


class StateRepresentationInterface(ABC):
    @abstractmethod
    def __getitem__(self, x):
        pass
