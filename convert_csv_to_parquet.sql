-- Convert all CSV files in join_test_hive to parquet.
-- To execute, run the following command:
-- datafusion-cli -f convert_csv_to_parquet.sql

-- dim table partitions
COPY (SELECT * FROM 'testdata/join_test_hive/dim/d_dkey=A/data.csv')
TO 'testdata/join_test_hive/dim/d_dkey=A/data.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/dim/d_dkey=B/data.csv')
TO 'testdata/join_test_hive/dim/d_dkey=B/data.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/dim/d_dkey=C/data.csv')
TO 'testdata/join_test_hive/dim/d_dkey=C/data.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/dim/d_dkey=D/data.csv')
TO 'testdata/join_test_hive/dim/d_dkey=D/data.parquet'
STORED AS PARQUET;

-- fact table partitions
COPY (SELECT * FROM 'testdata/join_test_hive/fact/f_dkey=A/data.csv')
TO 'testdata/join_test_hive/fact/f_dkey=A/data.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/fact/f_dkey=B/data.csv')
TO 'testdata/join_test_hive/fact/f_dkey=B/data.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/fact/f_dkey=B/data2.csv')
TO 'testdata/join_test_hive/fact/f_dkey=B/data2.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/fact/f_dkey=B/data3.csv')
TO 'testdata/join_test_hive/fact/f_dkey=B/data3.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/fact/f_dkey=C/data.csv')
TO 'testdata/join_test_hive/fact/f_dkey=C/data.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/fact/f_dkey=C/data2.csv')
TO 'testdata/join_test_hive/fact/f_dkey=C/data2.parquet'
STORED AS PARQUET;

COPY (SELECT * FROM 'testdata/join_test_hive/fact/f_dkey=D/data.csv')
TO 'testdata/join_test_hive/fact/f_dkey=D/data.parquet'
STORED AS PARQUET;
