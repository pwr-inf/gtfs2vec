"""Utils for processing data."""
from typing import Dict, List, Tuple, Union

import pandas as pd
from src.configurations import NORM_COLUMNS, USED_COLUMNS


def min_max_normalization_global(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    """Min-max normalization accross all rows in df and given columns.

    Args:
        df (pd.DataFrame): data
        columns (List): columns to normalize

    Returns:
        pd.DataFrame: normalized data
    """
    df[columns] = (df[columns] - df[columns].min().min()) / (df[columns].max().max() - df[columns].min().min())

    return df

def min_max_normalization_local(df: pd.DataFrame, columns: List, agg_column: str) -> pd.DataFrame:
    """Min-max normalization accross subsets of rows and given columns.

    Args:
        df (pd.DataFrame): data
        columns (List): columns to normalize
        agg_column (str): column to group rows for normalization

    Returns:
        pd.DataFrame: normalized data
    """
    for value in df[agg_column].unique():
        subset_df = df.loc[df[agg_column] == value, columns]
        df.loc[df[agg_column] == value, columns] = (
            (subset_df - subset_df.min().min()) / (subset_df.max().max() - subset_df.min().min())
        )

    return df


def normalize_data(
    df: pd.DataFrame, 
    column_groups: Union[Dict[str, List[int]], None], 
    agg_column: str,
    drop_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs min-max normalization both local and global.

    Args:
        df (pd.DataFrame): features
        column_groups (Union[Dict[str, List[int]], None]): dict with key is column name 
        and value is range of hours for column to run normalization on multiple columns
        agg_column (str): agregation column for local normalization
        drop_columns (List[str]): columns to drop before running normalization

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: local_df, global_df
    """
    local_df = df.drop(columns=drop_columns)
    global_df = df.drop(columns=drop_columns)

    if column_groups is None:
        norm_cols = NORM_COLUMNS
    else:
        norm_cols = column_groups

    for column, h_range in norm_cols.items():
        local_df = min_max_normalization_local(local_df, [f'{column}_{i}' for i in h_range], agg_column=agg_column)
        global_df = min_max_normalization_global(global_df, [f'{column}_{i}' for i in h_range])

    local_df = local_df[USED_COLUMNS]
    global_df = global_df[USED_COLUMNS]

    return local_df, global_df
