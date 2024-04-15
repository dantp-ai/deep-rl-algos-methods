from abc import ABC, abstractmethod

class StateRepresentationInterface(ABC):
    @abstractmethod
    def __getitem__(self, x):
        pass