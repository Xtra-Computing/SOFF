from abc import ABC, abstractclassmethod, abstractmethod
from collections.abc import MutableSequence
from typing import Type
from ...utils.arg_parser import BaseConfParser


class Node(ABC):
    """
    Base class for all algorithm nodes.
    Algorithm nodes must inherit this class
    """
    @abstractclassmethod
    def conf_parser(cls) -> Type[BaseConfParser]:
        """Override this method to specify main conf parser class"""
        raise NotImplementedError("Main config parser class mut be specified.")

    @abstractmethod
    def start_training(self, cfg: MutableSequence):
        """Override this method to start training"""
        raise NotImplementedError("This method must be overriden")

    @abstractmethod
    def cleanup(self):
        """Override this method to cleanup when training finished:"""
        raise NotImplementedError("Override this method to start training")
