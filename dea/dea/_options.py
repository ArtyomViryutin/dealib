__all__ = ["RTS", "Orientation"]

import enum


class BaseEnum(enum.Enum):
    @classmethod
    def get(cls, val):
        if isinstance(val, str):
            enums = {item.name: item.value for item in cls}
            if val not in enums:
                raise ValueError(
                    f"'{val}' is not a valid {cls.__name__}. Available values: {', '.join(enums.keys())}"
                )
            val = enums[val]
        return cls(val)

    def __str__(self):
        return self.name


class RTS(BaseEnum):
    vrs = 0
    crs = 1
    drs = 2
    irs = 3


class Orientation(BaseEnum):
    input = 0
    output = 1
