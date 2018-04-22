from pyspark import SparkContext, SparkConf
import numpy as np

PATH = 'results23_5.csv'


def map_values_1(row):
    fields = row.split(',')
    return fields[0], np.mean([fields[9], fields[13], fields[17], fields[22]])

if __name__ == "__main__":
    conf = SparkConf().setAppName('BSRP')
    sc = SparkContext(conf=conf)
    file_rdd = sc.textFile(PATH)
    first_line = file_rdd.filter(lambda x: 'mlp_layers' in x)  # Filter takes the untrue expression in
    no_header_rdd = file_rdd.subtract(first_line)  # Remove the header
    filtered_rdd = no_header_rdd.map(map_values_1)

    #print filtered_rdd.take(10)
