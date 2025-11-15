# db/create_table_from_csv.py
import pandas as pd
import sqlalchemy
import sys
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Synthetic_Transportation_Dataset_Expanded_v2.csv")

def infer_sql_type(pd_series):
    if pd.api.types.is_integer_dtype(pd_series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(pd_series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(pd_series):
        return "BOOLEAN"
    # fallback: text (and try to detect ISO timestamp)
    if pd.api.types.is_datetime64_any_dtype(pd_series):
        return "TIMESTAMP"
    # try parse datetime
    try:
        pd.to_datetime(pd_series.dropna().iloc[:50])
        return "TIMESTAMP"
    except Exception:
        return "TEXT"

def main():
    df = pd.read_csv(CSV_PATH, nrows=1000)
    # choose table name
    table_name = "training_data"
    # build CREATE TABLE
    cols = []
    for col in df.columns:
        sql_type = infer_sql_type(df[col])
        # sanitize column name
        col_name = col.strip().lower().replace(" ", "_")
        cols.append(f'"{col_name}" {sql_type}')

    create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} (\n  id SERIAL PRIMARY KEY,\n  ' + ",\n  ".join(cols) + "\n);"
    print("Generated CREATE statement:\n")
    print(create_stmt)
    # now push it to postgres using env vars or default
    # get connection from environment
    POSTGRES_USER = os.getenv("POSTGRES_USER", "devuser")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "devpass")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "devdb")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    engine_url = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    engine = sqlalchemy.create_engine(engine_url)

    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(create_stmt))
    print("Table created, now copying CSV (this requires CSV file accessible to the script).")

    # load full CSV and insert via pandas
    df_full = pd.read_csv(CSV_PATH)
    # sanitize column names
    df_full.columns = [c.strip().lower().replace(" ", "_") for c in df_full.columns]
    df_full.to_sql(table_name, engine, if_exists="append", index=False)
    print("CSV data loaded into table:", table_name)

if __name__ == "__main__":
    main()
