from pyspark.sql import SparkSession
from pyspark import SQLContext, SparkContext, SparkConf
from pyspark.sql.functions import concat, col, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.types import *
import json
import base64
import sys
from time import gmtime, strftime
from pyspark.sql.types import StringType
import csv
from pyspark.sql.types import StructField,StringType,IntegerType,StructType
from pyspark.sql.functions import format_number,dayofmonth,hour,dayofyear,month,year,weekofyear,date_format, minute
from pyspark.sql.functions import countDistinct, avg,stddev
from pyspark.sql.functions import format_number
#import tldextract
import string
import numpy as np, re, random
import zlib

########################################################################################################################
# Hieu Nguyen - DNS Tunneling - Technique: isolation forest - What I am doing here: I extract features from the data :)
# Date: April 20th, 2018
# Train, fit and finetune hyperparameters for both techniques
########################################################################################################################

# declare var >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
schema = StructType([StructField('url',StringType(),True)])

schema2 = StructType([StructField('queryUrl',StringType(),True), StructField('subdomain',StringType(),True), StructField('domain', StringType(), True),\
  StructField('suffix', StringType(), True), StructField('rootdomain', StringType(), True), StructField('subdomain_length', IntegerType(), True),\
  StructField('query_length', IntegerType(), True), StructField('labels', IntegerType(), True),StructField('alpha', IntegerType(), True),\
  StructField('numUniqueChar', IntegerType(), True), StructField('numerical', IntegerType(), True)])

# alphabetical library 'abcde...z'
alphabet = string.ascii_lowercase
punc1 = list('!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~')

# helper funcs:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# This function will only output valid dns queries
def getValidURL(url):
    temp = url.split(".")
    if len(temp) > 1:
        return url
    else:
        return ""
getValidURL_udf = udf(getValidURL, StringType())

def noSpecialChar(punc1, astr):
  numspec = 0
  for i in astr:
    if i in punc1:
      numspec +=1
  return numspec

def calculateCompression(seq):
    """calculates compression of the given sequence
    """
    try:
        temp = [chr((x/5)%128) for x in seq]
        comp = len(zlib.compress(''.join((temp))))/float(len(''.join(temp)))
    except ValueError:
        comp = 100
    return comp

def extractAllFeatures(alst):
  # NOTE that this function is not yet optimized. It is still using double loops :( Please don't judge :D
  list0 = []
  for x in alst:
    # for every query, extract the following:
    y = tldextract.extract(x) # beak the query into appropriate component:
    sub = y.subdomain # subdomain www
    dom = y.domain # domain google
    suf = y.suffix # com
    char = 0 # number of alphabetical character in a query
    num = 0 # number of numerical character in a query
    char1 = [] # append all alphabetical into this list
    lenUniqueChar = 0 # number of unique characters being used
    # If the query is a proper query, do this:
    if y.suffix != "" and y.subdomain != "":
      for i in x: # for every character in a query, do this:
        if i.lower() in alphabet: # if it is an alphabetical, then:
          char+=1 # increment char by 1
          char1.append(i) # append that character into char1 list
          lenUniqueChar = len(set(char1)) # get the total number of UNIQUE characters being used for that query
        num = len(x) - char - x.count(".") - x.count("-") # the number of numerical characters equals to this anything that is not alphabetical or - or .
    # So for each query in the list, create the following list:
    # example: ['www.google.com', 'wwww.hq2nguye.com'] ---> [['www.google.com', 'www', 'google', 'com'],['www.hq2nguye.com','www','hq2nguye','com']]
    # this list will then be used to construct a pyspark data frame. Then we can do the inner join back to the original data frame
    list0.append([x,sub.lower(), dom.lower(), suf.lower(), dom.lower() + "." + suf.lower(), len(sub), len(x), x.count(".")+1, char, lenUniqueChar, num])
  return list0

def get_num_specialChar(query_url):
    x = noSpecialChar(punc1, query_url)
    return x
udf_get_num_specialChar = udf(get_num_specialChar,IntegerType())

def subdomain_extract(subdomain):
  numchar = 0 # number of alphabetical chars
  numCom = 0 # number of numerical chars
  lenChar = 0 # number of unique chars
  Uchar = [] # append all alphabetical chars here
  lenUchar = 0 # we can calculate the number of unique chars being used
  charPerc = 1 # alphabetical percentage
  numPerc = 1 # alphabetical percentage
  for char in subdomain:
    if char in alphabet:
      numchar += 1
      Uchar.append(char)
      lenUchar = len(set(Uchar))
    numCom = len(subdomain) - numchar - subdomain.count(".") - subdomain.count("-")
    numPerc = np.round(float(numCom)/len(subdomain),3)
    charPerc = np.round(float(numchar)/len(subdomain),3)
  return [float(lenUchar), charPerc, numPerc, charPerc+numPerc]

def get_domain(query_url):
  y = tldextract.extract(query_url)
  x = subdomain_extract(y.subdomain)
  if y.suffix != "" and y.subdomain != "":
    return [y.subdomain, y.domain, y.suffix, y.domain + '.' + y.suffix, float(len(y.subdomain)), x[0], x[1], x[2], x[3]]
  else:
    return [0.0]*9

# Schema and UDF for get_domain output
schema2 = StructType([
  StructField("subdomain", StringType(), True),
  StructField("domain", StringType(), True),
  StructField("suffix", StringType(),True),
  StructField("rootdomain",StringType(),True),
  StructField("subdomain_length",IntegerType(),True),
  StructField("numUniqueChar", FloatType(), True),
  StructField("charPercent", FloatType(), True),
  StructField("numPercent", FloatType(),True),
  StructField("SumCompo",FloatType(),True)])

udf_get_domain = F.udf(get_domain, schema2)

# Processing begins >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def phase1(inputpath): # reads in the path
  dataDNS = spark.read.parquet(inputpath)
  print "Dataframe has been read ------------------>"
  return dataDNS

def phase2(dataDNS):
  date_event = str(dataDNS.select('iso_timestamp').take(1)[0][0][:10])
  dns = dataDNS.filter(dataDNS['type_dns'].isNotNull()) # remove all null
  
  # select relevant columns for investigation
  dns0 = dns.select('syslog_timestamp', 'msg', 'client_ip', 'client_port',F.lower(col('query_url')), 'type_dns', 'dns_server_ip', 'iso_timestamp')
  
  # remove traffics generated by servers
  dns1 = dns0.filter( ~(dns0['client_ip']==dns0['dns_server_ip']) )
  
  # investigate only relevant DNS types that are used for tunneling
  # note that type 'PRT' is not included because I could not find a good library to extract PRT dns queries.
  dns2 = dns1.filter( (dns1['type_dns']== 'A') | (dns1['type_dns']== 'AAAA') | (dns1['type_dns']== 'TXT') | (dns1['type_dns']=='CNAME'))

  # Filtering unwanted query_urls and parsing lower(query_url) into subdomain and domain
  dns3 = dns2.where("lower(query_url) not like '%.in-addr.arpa%' and lower(query_url) not like '%.rbc%' and lower(query_url) not like '%.royalbank%' and lower(query_url) not like '%.ip6.arpa%' and lower(query_url) not like '%.senderbase%' and lower(query_url) not like '%.spotify%' and lower(query_url) not like '%.mcafee%'")
  #dns3a = dns3.limit(10000).cache()
  dns4 = dns3.withColumn('numSpecChar', udf_get_num_specialChar(dns3['lower(query_url)']))
  dns5 = dns4.filter(dns4['numSpecChar'] == 0).withColumnRenamed('lower(query_url)', 'query_url')
  dns6 = dns5.withColumn('extracted_tld', udf_get_domain(F.regexp_extract(dns5['query_url'], '(.*://)?(www.)?(.*)', 3)))
  dns7 = dns6.select('syslog_timestamp', 'client_ip', 'query_url', 'type_dns', 'iso_timestamp',\
    'extracted_tld.subdomain', 'extracted_tld.domain', 'extracted_tld.suffix', 'extracted_tld.rootdomain', 'extracted_tld.subdomain_length',\
    'extracted_tld.numUniqueChar', 'extracted_tld.charPercent', 'extracted_tld.numPercent', 'extracted_tld.SumCompo')
  dns8 = dns7.filter(dns7['rootdomain'] != '0')
  dns8 = dns7.select('syslog_timestamp', 'client_ip', 'query_url', 'type_dns', 'iso_timestamp',\
    'subdomain', 'domain', 'suffix', 'rootdomain', 'subdomain_length', 'numUniqueChar', 'charPercent',\
    'numPercent', 'SumCompo')
  dns9 = dns8.filter( (dns8['SumCompo'] <= 1.00) & (dns8['charPercent'] > 0.30) ) # further filter down 
  dns10 = dns9.select('syslog_timestamp','client_ip','lower(query_url)', 'type_dns', 'iso_timestamp', 'subdomain', 'domain', 'suffix', 'rootdomain', 'subdomain_length','charPercent', 'numUniqueChar', 'numPercent')
  dns11 = dns10.withColumnRenamed('lower(query_url)','query_url').withColumn('timeEvent', F.lit(date_event))
  print "Phase 2 completing ---------------->>>>>"
  print dns11.show(5)

def phase3(dns11,outpath2):
  dns12 = dns11.groupBy('iso_timestamp','subdomain','rootdomain','suffix', 'type_dns').agg(F.countDistinct('client_ip'), F.count('rootdomain'),\
    F.avg('subdomain_length'), F.avg('numUniqueChar'), F.avg('charPercent'), F.avg('numPercent'))
  
  dns13 = dns12.select('iso_timestamp', 'subdomain' ,'rootdomain', 'suffix', 'type_dns',\
    F.round(col('count(DISTINCT client_ip)'),2).alias('NumUniqueIP').cast('Float'),\
    F.round(col('count(rootdomain)'),2).alias('Volume').cast('Float'),\
    F.round(col('avg(subdomain_length)'),2).alias('AvgSubdomainLength').cast('Float'),\
    F.round(col('avg(charPercent)'), 2).alias('AvgAlphaPercent').cast('Float'),\
    F.round(col('avg(numPercent)'),2).alias('AvgNumPercent').cast('Float'),\
    F.round(col('avg(numUniqueChar)'),2).alias('AvgUniqueChar').cast('Float'))

  dns13.write.parquet(outpath2)
  print "Phase 3 completing ---------------->>>>>"
  return 0

def phase4(dns11, outpath1):
  dns11 = dns10.groupBy('iso_timestamp','rootdomain','suffix', 'type_dns').agg(F.countDistinct('client_ip').cast('Float'), F.count('rootdomain').cast('Float'),\
    F.avg('subdomain_length'), F.avg('numUniqueChar'),F.avg('charPercent'),F.avg('numPercent'))
  dns12 = dns_main8.select('iso_timestamp', 'rootdomain', 'suffix', 'type_dns',\
    F.round(col('count(DISTINCT client_ip)'),2).alias('NumUniqueIP').cast('Float'),\
    F.round(col('count(rootdomain)'),2).alias('Volume').cast('Float'),\
    F.round(col('avg(subdomain_length)'),2).alias('AvgSubdomainLength').cast('Float'),\
    F.round(col('avg(charPercent)'), 2).alias('AvgAlphaPercent').cast('Float'),\
    F.round(col('avg(numPercent)'),2).alias('AvgNumPercent').cast('Float'),\
    F.round(col('avg(numUniqueChar)'),2).alias('AvgUniqueChar').cast('Float'))
  dns13 = final_test.filter(final_test['NumUniqueIP'] > 60)
  dns14 = dns_main9.withColumn('assumeBenign', F.lit(1))
  dns14.write.parquet(outpath1)
  print "Phase 4 Completed---------------------------->"
  return 0

def main():
  dataDNS = phase1(inputpath)
  dns11 = phase2(dataDNS)
  #phase3(dns11, outpath2)
  print "ALL DONE BOYSSS"

if __name__ == '__main__':
    conf_ = SparkConf().setAppName("Going_back_to_school")
    sc = SparkContext(conf=conf_)
    sc.addPyFile('hdfs:///prod/18010/app/UO50/analytics/python_library_egg/requests_file-1.4.3-py2.7.egg')
    sc.addPyFile('hdfs:///prod/18010/app/UO50/analytics/python_library_egg/tldextract-2.2.0-py2.7.egg')
    sc.addFile("hdfs:///prod/18010/app/UO50/analytics/StandardizeOutput/standardOutput.py")
    import tldextract
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName("HughJackman_iForestTest").config(conf=conf_).getOrCreate() 
    main()


# #########################################################################################################################
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SQLContext, SparkContext, SparkConf
from pyspark.sql.functions import concat, col, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.types import StructField,StringType,IntegerType,StructType, FloatType
from pyspark.sql.functions import format_number,dayofmonth,hour,dayofyear,month,year,weekofyear,date_format, minute
from pyspark.sql.functions import countDistinct, avg,stddev
from pyspark.sql.functions import format_number
import string
import random
import standardOutput

inputPath0 = '/prod/18010/app/UO50/analytics/Hieu/DNStunnel_Sept2017_Characteristics/Sept30/iForest/Set_2/Result_Set'
outputPath = '/prod/18010/app/UO50/analytics/Hieu/DNStunnel_Sept2017_Characteristics/As_April18th/iForest/Sept30.csv'

inputPath1 = '/prod/18010/app/UO50/analytics/Hieu/DNStunnel_Sept2017_Characteristics/Sept30/TestSet'

# udf that transform into json
udf_func1 = udf(lambda dnsQuery, Domain, type_dns, NumUniqueIP, Volume, subDomainRatio, Benign_subDomainRatio, AvgSubdomainLength, Benign_AvgSubdomainLength, AvgUniqueChar, Benign_AvgUniqueChar, AvgNumPercent, Benign_AvgNumPercent: '{' + '"SampleDnsQuery": \"{}\", "Domain": \"{}\", "typeDNS": \"{}\", "NumUniqueIP": {}, "Volume": {}, "subDomainRatio": {}, "Benign_subDomainRatio": {}, "AvgSubdomainLength": {}, "Benign_AvgSubdomainLength": {} , "AvgUniqueChar": {}, "Benign_AvgUniqueChar": {}, "AvgNumPercent": {}, "Benign_AvgNumPercent": {}'.format(dnsQuery, Domain, type_dns, NumUniqueIP, Volume, subDomainRatio, Benign_subDomainRatio, AvgSubdomainLength, Benign_AvgSubdomainLength, AvgUniqueChar, Benign_AvgUniqueChar, AvgNumPercent, Benign_AvgNumPercent) + '}')

udf_func2 = F.udf(lambda x: ", ".join(random.sample(x, min(3, len(x)))))

def transform(inputPath0,inputPath1):
  df0 = spark.read.parquet(inputPath0)
  df1 = spark.read.parquet(inputPath1)
  # group df1 by rootdomain and agg collect_set the query url
  df1_a = df1.groupBy('rootdomain').agg(udf_func2(F.collect_set('queryUrl')))
  # rename root domain to root b4 joining back
  df1_b = df1_a.withColumnRenamed('rootdomain','root')
  # inner join with df0
  df0_a = df0.join(df1_b, df0['rootdomain'] == df1_b['root'],'inner')
  df0_b = df0_a.drop('root').withColumnRenamed('<lambda>(collect_set(queryUrl, 0, 0))', 'dnsQuery')
  df0_c = df0_b.withColumnRenamed('timeEvent', 'iso_timestamp').withColumnRenamed('rootdomain','Domain').withColumn('Benign_AvgSubdomainLength', F.lit(float(5.770))).withColumn('Benign_subDomainRatio', F.lit(float(0.013)))\
    .withColumn('Benign_AvgNumPercent', F.lit(float(0.016))).withColumn('Benign_AvgUniqueChar',F.lit(float(10.861)))
  df0_d = df0_c.withColumn('Evidence', udf_func1('dnsQuery','Domain', 'type_dns', 'NumUniqueIP', 'Volume','subDomainRatio', 'Benign_subDomainRatio', 'AvgSubdomainLength', 'Benign_AvgSubdomainLength', 'AvgUniqueChar', 'Benign_AvgUniqueChar', 'AvgNumPercent', 'Benign_AvgNumPercent'))
  df0_e = df0_d.select('iso_timestamp', 'Domain', 'Evidence')
  #df0_c = df0_b.withColumnRenamed('timeEvent', 'iso_timestamp').withColumnRenamed('rootdomain','Domain')
  #df0_d = df0_c.withColumn('Evidence', udf_func1('dnsQuery','Domain', 'type_dns', 'NumUniqueIP', 'Volume', 'subDomainRatio', 'AvgSubdomainLength', 'AvgUniqueChar', 'AvgNumPercent'))
  print "DONE DEAL MAN! ------------------>>>"
  return df0_e

def main():
  df0_e = transform(inputPath0,inputPath1)
  standardOutput.std_infoblox(df0_e, outputPath, '2017-09-30', '2017-09-30','2017-09-30','DNStunnel_Sept2017_iForest','Infoblox')
  print "ALL DONE DEAL MAN! ------------------>>>"
  return 0
  
if __name__ == '__main__':
    conf_ = SparkConf().setAppName("My_eyes_are_big")
    sc = SparkContext(conf=conf_)
    sc.addFile("hdfs:///prod/18010/app/UO50/analytics/StandardizeOutput/standardOutput.py")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName("HughJackman_iForestTest").config(conf=conf_).getOrCreate() 
    main()
