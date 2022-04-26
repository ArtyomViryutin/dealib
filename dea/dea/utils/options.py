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
    """
    Class to configure returns to scale assumption.

    **vrs** - Variable returns to scale

    **crs** - Constant returns to scale

    **drs** - Decreasing returns to scale

    **irs** - Increasing returns to scale
    """

    vrs = 0
    crs = 1
    drs = 2
    irs = 3


class Orientation(BaseEnum):
    """
    Class to configure efficiency orientation.

    **input** - input efficiency

    **output** - output efficiency
    """

    input = 0
    output = 1
