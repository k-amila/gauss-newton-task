import pandas as pd


def get_data(filename, project_id, size):
    df = pd.read_csv(filename)
    df.x = df.x / 85000
    df8 = df[df.project_id == project_id]

    x = df8.x[0:size].tolist()
    y = df8.y[0:size].tolist()
    return x, y