from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("StockMarketML").getOrCreate()
print("Spark session started...")

# Load data
input_path = "gs://ankit-00-bucket/output/Cleaned_data"
print(f"Loading data from: {input_path}")

df = spark.read.csv(input_path, header=True, inferSchema=False)

# Cast numeric columns to DoubleType
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
for column in numeric_columns:
    df = df.withColumn(column, col(column).cast(DoubleType()))

# Convert Date to timestamp
df = df.withColumn("Date", to_timestamp("Date", "yyyy-MM-dd"))  

# Confirm schema
df.printSchema()

# Define features and label
feature_cols = ["Open", "High", "Low", "Volume"]
target_col = "Close"

# Assemble features (skip rows with nulls)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
df = assembler.transform(df)

# Prepare final DataFrame for training
model_df = df.select("features", col(target_col).alias("label")).na.drop()

# Split data
train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LinearRegression(featuresCol="features", labelCol="label", regParam=0.1)
lr_model = lr.fit(train_df)

# Evaluate
print("Starting evaluation...")
print("Schema of test_df:")
test_df.printSchema()
print("Sample test data:")
test_df.show(5)
test_results = lr_model.evaluate(test_df)
test_results.predictions.show()

try:
    # your code that might throw an error
    test_results = lr_model.evaluate(test_df)
except Exception as e:
    print("Evaluation failed:", e)
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R2: {test_results.r2}")
print("Test set count:", test_df.count())


# Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Closing Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# Visulaizing the Results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.xlabel('Actual Closing Price')
plt.ylabel('Predicted Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.grid(True)
plt.show()

# Print model coefficients
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_})
print(coeff_df)

spark.stop()
print("Spark job completed successfully ✅")
