#!/usr/bin/env python
# coding: utf-8
pip install pyspark
import pyspark
import pandas as pd

pd.read_csv("terrorist_data_all.csv")

from pyspark.sql import SparkSession

# create a SparkSession
spark = SparkSession.builder.appName("Clustering").getOrCreate()

# read the CSV file into a DataFrame
data = spark.read.format("csv").option("header", "true").load("terrorist_data_all.csv")

from pyspark.ml.feature import VectorAssembler, StringIndexer

# create a StringIndexer for each column with string values
actor_name_indexer = StringIndexer(inputCol="ActorName", outputCol="ActorNameIndex")
actor_type_indexer = StringIndexer(inputCol="ActorType", outputCol="ActorTypeIndex")
affiliation_to_indexer = StringIndexer(inputCol="AffiliationTo", outputCol="AffiliationToIndex")
affiliation_start_date_indexer = StringIndexer(inputCol="AffiliationStartDate", outputCol="AffiliationStartDateIndex")
affiliation_end_date_indexer = StringIndexer(inputCol="AffiliationEndDate", outputCol="AffiliationEndDateIndex")
aliases_indexer = StringIndexer(inputCol="Aliases", outputCol="AliasesIndex")

# fit the StringIndexer models to the data
actor_name_indexer_model = actor_name_indexer.fit(data)
actor_type_indexer_model = actor_type_indexer.fit(data)
affiliation_to_indexer_model = affiliation_to_indexer.fit(data)
affiliation_start_date_indexer_model = affiliation_start_date_indexer.fit(data)
affiliation_end_date_indexer_model = affiliation_end_date_indexer.fit(data)
aliases_indexer_model = aliases_indexer.fit(data)

# transform the data using the StringIndexer models
data = actor_name_indexer_model.transform(data)
data = actor_type_indexer_model.transform(data)
data = affiliation_to_indexer_model.transform(data)
data = affiliation_start_date_indexer_model.transform(data)
data = affiliation_end_date_indexer_model.transform(data)
data = aliases_indexer_model.transform(data)

# define the columns that contain the features
feature_cols = ["ActorNameIndex", "ActorTypeIndex", "AffiliationToIndex", "AffiliationStartDateIndex", "AffiliationEndDateIndex", "AliasesIndex"]

# create a VectorAssembler object to assemble the features into a vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# transform the data to create the feature vectors
feature_vectors = assembler.transform(data).select("features")

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

# create a SparkSession
spark = SparkSession.builder.appName("Clustering").getOrCreate()

# read in the data
df = spark.read.csv("terrorist_data_all.csv", header=True, inferSchema=True)

# select the relevant columns for clustering
data = df.select("ActorType", "AffiliationTo", "Aliases")

# perform one-hot encoding on categorical variables
from pyspark.ml.feature import OneHotEncoder, StringIndexer
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(data) for column in ["ActorType", "AffiliationTo", "Aliases"]]
encoders = [OneHotEncoder(inputCol=column+"_index", outputCol=column+"_vec") for column in ["ActorType", "AffiliationTo", "Aliases"]]
pipeline = Pipeline(stages=indexers + encoders)
data_encoded = pipeline.fit(data).transform(data)

# assemble the feature vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["ActorType_vec", "AffiliationTo_vec", "Aliases_vec"], outputCol="features")
feature_vectors = assembler.transform(data_encoded).select("features")

# create a KMeans object with k=5 clusters
kmeans = KMeans().setK(5)

# fit the KMeans model to the feature vectors
model = kmeans.fit(feature_vectors)

# get the cluster labels for each data point
labels = model.transform(feature_vectors).select("prediction").rdd.map(lambda x: x[0]).collect()

# stop the SparkSession
spark.stop()

import networkx as nx
import matplotlib.pyplot as plt

# create a new graph
graph = nx.Graph()

# add the nodes with the cluster labels as the node labels
for i, label in enumerate(labels):
    graph.add_node(i, label=label)

# add the edges based on the connections between the communities
# this will depend on the structure of your data
# you may need to modify this code to fit your data
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        # modify this condition to fit your data
        if labels[i] == labels[j]:
            graph.add_edge(i, j)

# draw the graph
nx.draw(graph, with_labels=True)
plt.show()
