import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf,
    year,
    month,
    dayofmonth,
    hour,
    weekofyear,
    dayofweek,
    monotonically_increasing_id,
)

config = configparser.ConfigParser()
config.read("dl.cfg")

os.environ["AWS_ACCESS_KEY_ID"] = config["conf"]["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = config["conf"]["AWS_SECRET_ACCESS_KEY"]


def create_spark_session():
    spark = SparkSession.builder.config(
        "spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0"
    ).getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Saving Songs Data as Parquet after Modeling
    """
    # read song data file
    df = spark.read.json(f"{input_data}/song_data/A/A/A/*.json")

    # extract columns to create songs table
    songs_table = df.select(
        ["song_id", "title", "artist_id", "year", "duration"]
    ).dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").parquet(
        f"{output_data}/songs", mode="overwrite"
    )

    # extract columns to create artists table
    artists_table = df.select(
        [
            "artist_id",
            "artist_name",
            "artist_location",
            "artist_latitude",
            "artist_longitude",
        ]
    ).dropDuplicates()

    # write artists table to parquet files
    artists_table.write.parquet(f"{output_data}/logs", mode="overwrite")


def process_log_data(spark, input_data, output_data):
    """
    Saving Logs Data as Parquet after Modeling
    """
    # read log data file
    df = spark.read.json(f"{input_data}/log_data/*/*/*.json")

    # filter by actions for song plays
    df = df.filter("page = 'NextSong'")

    # extract columns for users table
    users_table = df.select(
        ["userId", "firstName", "lastName", "gender", "level"]
    ).dropDuplicates()

    # write users table to parquet files
    users_table.write.parquet(f"{output_data}/users", mode="overwrite")

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x / 1000))
    df = df.withColumn("start_time", get_datetime("ts"))

    time_table = df.select("start_time")

    # extract columns to create time table
    time_table = time_table.withColumn("hour", hour("start_time"))
    time_table = time_table.withColumn("day", dayofmonth("start_time"))
    time_table = time_table.withColumn("week", weekofyear("start_time"))
    time_table = time_table.withColumn("month", month("start_time"))
    time_table = time_table.withColumn("year", year("start_time"))
    time_table = time_table.withColumn("weekday", dayofweek("start_time"))

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(
        f"{output_data}/times", mode="overwrite"
    )

    # read in song data to use for songplays table
    song_df = spark.read.json(f"{input_data}/song_data/A/A/A/*.json")

    # extract columns from joined song and log datasets to create songplays table
    df = df.orderBy("ts")
    df = df.withColumn("songplay_id", monotonically_increasing_id())

    song_df.createOrReplaceTempView("songs")
    df.createOrReplaceTempView("events")

    songplays_table = spark.sql(
        """
        SELECT e.songplay_id,
               e.start_time,
               e.userId AS user_id,
               e.level,
               s.song_id,
               s.artist_id,
               e.sessionId AS session_id,
               e.location,
               e.userAgent AS user_agent,
               year(e.start_time) AS YEAR,
               month(e.start_time) AS MONTH
        FROM EVENTS e
        LEFT JOIN songs s ON e.song = s.title
        AND e.artist = s.artist_name
        """
    )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(
        f"{output_data}/songplays", partitionBy=["year", "month"], mode="overwrite"
    )


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3://dlsavingdata/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
