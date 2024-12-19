from dataclasses import dataclass

from trl.core import flatten_dict


@dataclass
class BaseConfig:
    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
