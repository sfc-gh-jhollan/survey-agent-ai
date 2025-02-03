import modin.pandas as pd

# Import the Snowpark plugin for modin.
import snowflake.snowpark.modin.plugin

# Create a Snowpark session with a default connection.
from snowflake.snowpark.session import Session

session = Session.builder.config("connection_name", "pm").getOrCreate()
session.use_database("JEFFHOLLAN_DEMO")
session.use_schema("SURVEY")

# List of CSV files to process
csv_files = [
    "patients.csv",
    "survey_analytics.csv",
    "survey_attempts.csv",
    "survey_costs.csv",
    "survey_responses.csv",
    "visits.csv",
    "cost_by_week.csv",
]

# Process each CSV file
for csv_file in csv_files:
    table_name = csv_file.split(".")[0].upper()
    df = pd.read_csv(f"data/{csv_file}")
    # uppercase column names
    df.columns = df.columns.str.upper()
    session.write_pandas(
        df, table_name=table_name, overwrite=True, auto_create_table=True
    )

# Read and print the data from the tables
for csv_file in csv_files:
    table_name = csv_file.split(".")[0].upper()
    df = pd.read_snowflake(f"JEFFHOLLAN_DEMO.SURVEY.{table_name}")
    print(f"{table_name} DATA")
    print(df)
