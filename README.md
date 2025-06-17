## Project Overview

# Big-Data-Analytics
This project demonstrates the creation of a scalable data pipeline for stock market data analysis using Big Data tools on Google Cloud Platform (GCP). The pipeline leverages Google Dataproc for distributed data processing, Hive queries for data warehousing and querying, and PySpark for data transformation and analysis.
# Big Data Stock Market Analysis with Google Dataproc


## Objectives

- Ingest and process large volumes of stock market data.
- Build a scalable data pipeline using Google Dataproc.
- Perform data querying with Hive on Dataproc clusters.
- Analyze and process data using PySpark on GCP.

## Tools & Technologies

- **Google Cloud Platform (GCP)**
- **Google Dataproc**: Managed Spark and Hadoop service for big data processing.
- **Hive**: SQL-like querying for structured data analysis.
- **PySpark**: Python API for Spark, used for distributed data processing and analytics.

## Pipeline Workflow

1. **Data Ingestion:**  
   Load raw stock market data into Google Cloud Storage.

2. **Cluster Setup:**  
   Provision a Google Dataproc cluster for running Hive and PySpark jobs.

3. **Data Querying (Hive):**  
   Use Hive queries on Dataproc to perform ETL (Extract, Transform, Load) operations and organize data into tables.

4. **Data Analysis (PySpark):**  
   Use PySpark scripts on Dataproc to perform advanced analytics, such as trend detection, moving averages, or anomaly detection on stock prices.

5. **Result Storage & Visualization:**  
   Store results in a suitable format (e.g., CSV, Parquet) in Google Cloud Storage for further visualization or reporting.

## Getting Started

1. Clone the repository.
2. Set up your GCP project and enable Dataproc.
3. Upload sample stock market data to Google Cloud Storage.
4. Follow the provided Hive and PySpark scripts to process and analyze the data.

## Folder Structure

- `/hive-scripts`: Hive queries for ETL and data warehousing.
- `/pyspark-scripts`: PySpark scripts for data analysis.
- `/data-samples`: Sample stock market data files.
- `/docs`: Documentation and pipeline diagrams.
