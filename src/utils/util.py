def map_scalar_to_cell(action):
    row = (action // 3)
    column = (action % 3)
    return row, column

def map_cell_to_scalar(x, y):
    scalar = x * 3
    scalar += y
    return scalar