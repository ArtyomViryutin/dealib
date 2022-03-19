__all__ = ["Orientation", "RTS", "Model"]

from enum import Enum, auto


class BaseEnum(Enum):
    @classmethod
    def get(cls, val):
        if not isinstance(val, cls):
            enums = {item.name: item.value for item in cls}
            if val not in enums:
                raise ValueError(
                    f"'{val}' is not a valid {cls.__name__}. Available values: {', '.join(enums.keys())}"
                )
            val = enums[val]
        return cls(val)

    def __str__(self):
        return self.name


class Orientation(BaseEnum):
    input = auto()
    output = auto()


class RTS(BaseEnum):
    vrs = auto()
    crs = auto()
    drs = auto()
    irs = auto()


class Model(BaseEnum):
    envelopment = auto()
    multiplier = auto()
