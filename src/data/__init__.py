from importlib.resources import files
import pandas as pd

_timeseries_path = files("data").joinpath("processed/timeseries_corrected.csv")
timeseries = pd.read_csv(_timeseries_path)
# excel_results = pd.read_csv(_path_to_excel_reference, delimiter=";", decimal=",")
timeseries["date"] = pd.to_datetime(timeseries["date"])
timeseries = timeseries[~pd.isnull(timeseries["discharge"])]