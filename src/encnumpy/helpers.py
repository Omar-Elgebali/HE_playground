from math import ceil
from .types import Any, Tuple, Type, Shape, Union, List

_ceil_step = lambda idx, step: ceil(abs(idx) / step) * step


def determine_shapes(shape: Shape) -> Tuple[Shape, Shape]:
    if not shape:  # Empty tuple case
        new_shape = (None, None)
        new_shape_actual = (1, 1)
    elif len(shape) == 1:  # Single-element tuple case
        (dim,) = shape  # Unpack the single value
        new_shape = (dim, None)
        new_shape_actual = (1, dim)
    else:  # Multi-element tuple case
        new_shape = new_shape_actual = shape
    return new_shape, new_shape_actual


def can_broadcast(shape_a: Shape, shape_b: Shape) -> bool:
    """
    Checks if two shapes are compatible for broadcasting in NumPy operations.

    :param shape_a: Tuple representing the shape of the first array.
    :param shape_b: Tuple representing the shape of the second array.
    :return: True if the two shapes are broadcast-compatible, False otherwise.
    """

    # Pad the smaller shape with ones to match the larger shape's length
    len_a, len_b = len(shape_a), len(shape_b)
    if len_a < len_b:
        shape_a = (1,) * (len_b - len_a) + shape_a
    elif len_b < len_a:
        shape_b = (1,) * (len_a - len_b) + shape_b

    # Reverse shapes to align from the right (trailing dimensions matter most)
    rev_a, rev_b = shape_a[::-1], shape_b[::-1]

    # Iterate over the aligned dimensions
    for dim_a, dim_b in zip(rev_a, rev_b):
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    # Handle special case for zero-size dimensions
    if 0 in shape_a or 0 in shape_b:
        return all(dim == 0 for dim in shape_a) == all(dim == 0 for dim in shape_b)

    return True

def get_ndim(matrix: Any) -> int:
    """
    Recursively determines the number of dimensions of a nested list or array-like structure.
    Args:
        matrix (Any): The nested list or array-like structure.
    Returns:
        int: The number of dimensions.
    """
    ndim = 0
    while isinstance(matrix, list):
        ndim += 1
        if not matrix:  # empty list
            break
        matrix = matrix[0]
    return ndim

def get_deepest_type(obj: Any) -> Type:
    """
    Recursively determines the type of the deepest element in a nested list or array-like structure.
    Args:
        obj (Any): The nested list or array-like structure.
    Returns:
        Type: The type of the deepest element.
    """
    try:
        return get_deepest_type(obj[0])
    except Exception:
        t = type(obj)
        # Map NumPy scalar types to Python built-in types
        if t.__module__ == 'numpy':
            if 'int' in t.__name__:
                return int
            if 'float' in t.__name__:
                return float
            if 'bool' in t.__name__:
                return bool
        return t

def convert_bools_to_ints(obj: Any) -> Any:
    """
    Recursively converts boolean values in a nested list or array-like structure to integers.
    Args:
        obj (Any): The nested list or array-like structure.
    Returns:
        Any: The modified structure with booleans converted to integers.
    """
    try:
        return [convert_bools_to_ints(item) for item in obj]
    except TypeError:
        return int(obj)


def is_rectangular_nd(matrix: Any) -> bool:
    """
    Recursively checks whether a nested list is non-jagged (i.e., rectangular) in all dimensions.

    Args:
        matrix (Any): The nested list to check.

    Returns:
        bool: True if all dimensions are rectangular, False otherwise.
    """
    if not isinstance(matrix, list):
        return True  # Base case: scalar value

    if not matrix:
        return True  # Empty list is considered rectangular

    # All elements must be of the same type (either all lists or all scalars)
    first_elem_is_list = isinstance(matrix[0], list)

    for item in matrix:
        if isinstance(item, list) != first_elem_is_list:
            return False  # Mixed types at the same level

    if first_elem_is_list:
        # All sublists must be of the same length
        expected_len = len(matrix[0])
        for sublist in matrix:
            if len(sublist) != expected_len:
                return False
            if not is_rectangular_nd(sublist):
                return False

    return True


def format_nd_list(arr: Any, level: int = 0) -> str:
    """
    Formats a nested list or array-like structure into a string representation with proper indentation.
    Args:
        arr (Any): The nested list or array-like structure to format.
        level (int): The current indentation level (used for recursive calls).
    Returns:
        str: The formatted string representation of the nested structure.
    """
    indent = "  " * level
    next_indent = "  " * (level + 1)

    if not isinstance(arr, list) or not arr:
        return str(arr)

    if isinstance(arr[0], list):
        if isinstance(arr[0][0], list):
            # 3D or higher: group inner blocks with outer brackets
            inner_blocks = [format_nd_list(sub, level + 1) for sub in arr]
            joined = ",\n\n".join(inner_blocks)
            return indent + "[\n" + joined + "\n" + indent + "]"
        else:
            # 2D block: format each row with proper alignment
            formatted_rows = []
            for row in arr:
                formatted_row = " ".join(
                    (
                        f"{float(x):4.1f}".rstrip("0").rstrip(".") + "."
                        if float(x).is_integer()
                        else f"{float(x)}"
                    )
                    for x in row
                )
                formatted_rows.append(next_indent + "[" + formatted_row + "]")
            return indent + "[\n" + "\n".join(formatted_rows) + "\n" + indent + "]"
    else:
        # 1D list
        formatted_line = " ".join(
            (
                f"{float(x):4.1f}".rstrip("0").rstrip(".") + "."
                if float(x).is_integer()
                else f"{float(x)}"
            )
            for x in arr
        )
        return indent + "[" + formatted_line + "]"


def cut_shape(shape: Shape, axis: int, keepdims: bool) -> Tuple[Shape, Shape]:
    """
    Cuts the shape of a matrix along a specified axis.
    Args:
        shape (Shape): The original shape of the matrix.
        axis (int): The axis along which to cut the shape.
        keepdims (bool): Whether to keep the dimensions of the original shape.
    Returns:
        Tuple[Shape, Shape]: The new shape after cutting.
    """
    if keepdims:
        new_shape = shape[:axis] + (1,) + shape[axis + 1 :]
    else:
        new_shape = shape[:axis] + shape[axis + 1 :]

    # If the new shape is empty, return the original shape
    if not new_shape:
        new_shape_visual = (None, None)
        new_shape_actual = (1, 1)
    elif len(new_shape) == 1:
        new_shape_visual = (new_shape[0], None)
        new_shape_actual = (1, new_shape[0])
    else:
        new_shape_visual = new_shape
        new_shape_actual = new_shape
    return new_shape_visual, new_shape_actual


def broadcast_requirement(
    x1: Shape, x2: Shape, is_matmul: bool = False
) -> Tuple[Tuple[bool, bool], Shape, Shape, Shape, Shape]:
    """
    Determines the broadcast requirement and resulting shape for two input shapes.

    Args:
        x1 (Shape): The first input shape.
        x2 (Shape): The second input shape.
        is_matmul (bool): Whether this is for matrix multiplication; if so, cuts off the last two dimensions and appends matrix multiplication dimensions.
    Returns:
        Tuple:
            - (bool, bool): Whether the left and right shapes need broadcasting.
            - new_shape_1 (Shape): The new shape for the first input after broadcasting.
            - new_shape_2 (Shape): The new shape for the second input after broadcasting.
            - padded_batch_shape_1 (Shape): The padded batch shape for the first input.
            - padded_batch_shape_2 (Shape): The padded batch shape for the second input.
    Raises:
        ValueError: If the shapes are incompatible for broadcasting.
    """
    if is_matmul:
        if len(x1) < 2 or len(x2) < 2:
            raise ValueError("For matmul, both shapes must have at least 2 dimensions.")

        batch_x1, mat_x1 = x1[:-2], x1[-2:]
        batch_x2, mat_x2 = x2[:-2], x2[-2:]
    else:
        batch_x1, mat_x1 = x1, ()
        batch_x2, mat_x2 = x2, ()

    original_len_x1 = len(batch_x1)
    original_len_x2 = len(batch_x2)

    max_len = max(original_len_x1, original_len_x2)
    padded_x1 = (1,) * (max_len - original_len_x1) + batch_x1
    padded_x2 = (1,) * (max_len - original_len_x2) + batch_x2

    result_shape = []
    need_left = len(padded_x1) > original_len_x1
    need_right = len(padded_x2) > original_len_x2

    for dim1, dim2 in zip(padded_x1, padded_x2):
        if dim1 == dim2:
            result_shape.append(dim1)
        elif dim1 == 1:
            result_shape.append(dim2)
            need_left = True
        elif dim2 == 1:
            result_shape.append(dim1)
            need_right = True
        else:
            raise ValueError(f"Incompatible dimensions: {dim1} vs {dim2}")

    # Construct new shapes
    new_shape_1 = result_shape + list(mat_x1)
    new_shape_2 = result_shape + list(mat_x2)

    return (
        (need_left, need_right),
        tuple(new_shape_1),
        tuple(new_shape_2),
        (*padded_x1, *mat_x1),
        (*padded_x2, *mat_x2),
    )
