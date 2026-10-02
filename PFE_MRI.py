from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

#Création de la session Spark

spark = SparkSession.builder \
    .appName("PFE_MRI") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.hadoop.hadoop.home", "/home/daniel/hadoop") \
    .getOrCreate()


#Chargement des images depuis HDFS

df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load("hdfs://localhost:9000/PFE/Training/")

print("Nombre d'images chargées :", df.count())
df.show(5)


#Extraction du label

def extract_label(path):

    return path.split("/")[-2]

label_udf = udf(extract_label, StringType())

df = df.withColumn("label", label_udf(col("path")))

df.select("path", "label").show(10)


#Sauvegarde du dataset propre en Parquet dans HDFS

df.write.mode("overwrite").parquet("hdfs:///PFE/Net_Training/")

print("Dataset nettoyé sauvegardé dans HDFS.")
