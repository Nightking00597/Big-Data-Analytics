from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, avg
from pyspark.sql.window import Window

# 1. Initialize Spark session
spark = SparkSession.builder.appName("StockDataWrangling").getOrCreate()

# 2. Load stock data from GCS
file_path = "gs://ankit-00-bucket/nyse/"
df_raw = spark.read.option("header", True).option("inferSchema", True).csv(file_path)

# 3. Drop rows where Date is null & cast Date to timestamp
df = df_raw.filter(df_raw.Date.isNotNull())
df = df.withColumn("Date", col("Date").cast("timestamp")).orderBy("Date")

# Debug: Initial row count
initial_count = df.count()
print(f"Initial row count (after filtering Date): {initial_count}")
df.show(5, truncate=False)

# 4. Define window spec for lag & moving averages
window = Window.orderBy("Date")

# 5. Add lag features
df = df.withColumn("Prev_Close", lag("Close", 1).over(window))
df = df.withColumn("Prev_Volume", lag("Volume", 1).over(window))

# 6. Add moving average features
df = df.withColumn("MA_5", avg("Close").over(window.rowsBetween(-4, 0)))
df = df.withColumn("MA_10", avg("Close").over(window.rowsBetween(-9, 0)))

# 7. Drop rows with nulls in important columns (including lag/MA)
cols_to_check = ['Low', 'Open', 'High', 'Close', 'Adjusted Close', 'Volume', 'Prev_Close', 'Prev_Volume', 'MA_5', 'MA_10']
df_clean = df.dropna(subset=cols_to_check)

# Debug: Row counts before and after cleaning
print(f"Rows before dropna: {df.count()} → after dropna: {df_clean.count()}")

# 8. Save final cleaned data to GCS
output_path = "gs://ankit-00-bucket/output/Cleaned_data"
df_clean.write.mode("overwrite").option("header", True).csv(output_path)
print(f"✅ Saved cleaned data to: {output_path}")

# 9. Stop Spark session
spark.stop()
