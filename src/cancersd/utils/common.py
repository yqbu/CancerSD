import pickle
import copy
from typing import Any


def deepcopy(obj: Any, max_protocol: bool = True, fallback_to_copy: bool = True) -> Any:
    """
    achieve fast deep copy through pickle serialization

    :param obj: the object to be copied
    :param max_protocol: whether to use the most efficient pickle protocol (default True)
    :param fallback_to_copy: when pickle fails, whether to roll back to copy.deepcopy (default True)

    :return: the new object after a deepcopy

    :raises TypeError: when the object cannot be pickled and fallback_to_copy=False
    """
    try:
        protocol = pickle.HIGHEST_PROTOCOL if max_protocol else None
        return pickle.loads(pickle.dumps(obj, protocol=protocol))
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        if fallback_to_copy:
            return copy.deepcopy(obj)
        else:
            raise TypeError(
                f'Object {obj} cannot be pickled and fallback_to_copy=False'
            )