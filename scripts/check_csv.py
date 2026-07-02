import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', 1024)


if __name__ == "__main__":
    path = "D:\\seaf\\mlruns\\52\\64cba73d81154ed4acf712d94511fe81\\artifacts\\snapshots\\model_mlp\\model_mlp_in_2020-05-18.csv"
    df = pd.read_csv(path, compression='gzip')

    check_dates = ['2020-01-29', '2020-01-30', '2020-02-03', '2020-02-04', '2020-02-05']
    for date in check_dates:
        print(f"Checking date: {date}")
        df_date = df[df['key'] == date]
        print(df_date.tail(10))
        print("\n")
