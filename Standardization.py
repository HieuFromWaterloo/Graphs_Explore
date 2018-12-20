import base64
from pyspark.sql import SparkSession
from pyspark import SQLContext, SparkContext, SparkConf
from pyspark.sql.functions import concat, col, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.types import *
import datetime

# Global Variables
today = str(datetime.date.today())

# NOTE <><><><><><><><><><><><><><>
# startPeriod ----> the start date that you ran your model on
# endPeriod ----> the end date that you ran your model up to
# df -----> your output df
# outpath -----> where you would like to save the csv output
# datasource ----> name of the data source being used for the use case. ex: infoblox
# analytics -------> name of your use case; The name must follow this format as requested by Jamie: useCaseName_dataSource_technique ex. DNStunneling_Infoblox_iforest
# baseSeverity ---------> confirm with the threat team about this before you pass your final evidence into the standardization function


def encode64(s):
    return base64.b64encode(s.encode('utf-8'))
encode_udf = udf(encode64, StringType())

def std_palo(df, outpath, rundate, startPeriod, endPeriod, analytics, datasource):
	# PLEASE INCLUDE THE 'msg_type' column for palo alto!
	if df.select('msg_type').limit(1).rdd.flatMap(lambda x: x).collect()[0] == 'TRAFFIC' or df.select('msg_type').limit(1).rdd.flatMap(lambda x: x).collect()[0] == 'THREAT': # this emphasize the log type for PaloAlto
		listOfCols0 = ['iso_timestamp', 'confidence_level', 'source_ip', 'source_port', 'source_user', 'Domain','server_name', 'SRC DNS/NETBIOS NAME', 'destination_port','DST DNS/NETBIOS NAME', 'destination_ip']
		listOfCols1 = df.columns
	
		for col in listOfCols0:
			if col not in listOfCols1:
				df = df.withColumn(col, F.lit("Null"))

		df1 = df.withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('source_user', 'User').withColumnRenamed('source_ip', 'SRC IP')\
			.withColumnRenamed('source_port', 'SRC PORT').withColumnRenamed('destination_ip','DST IP').withColumnRenamed('destination_port', 'DST PORT').withColumnRenamed('server_name', 'Server Name')

		df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

		df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

		df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')
		print "TRANSFORMING PALO ALTO DATAFRAME COMPLETED ------------------------------------->"
		
	else: #if log is 'CONFIG'
		listOfCols0 = ['iso_timestamp', 'confidence_level', 'host', 'SRC PORT', 'source_user', 'Domain','server_name', 'SRC DNS/NETBIOS NAME', 'destination_port','DST DNS/NETBIOS NAME', 'destination_ip']
		# Existing domain column??
		listOfCols1 = df.columns
	
		for col in listOfCols0:
			if col not in listOfCols1:
				df = df.withColumn(col, F.lit("Null"))

		df1 = df.withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('source_user', 'User').withColumnRenamed('host', 'SRC IP')\
			.withColumnRenamed('destination_ip','DST IP').withColumnRenamed('destination_port', 'DST PORT').withColumnRenamed('server_name', 'Server Name')

		df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

		df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

		df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')
		print "TRANSFORMING PALO ALTO DATAFRAME COMPLETED ------------------------------------->"

	return 0

def std_infoblox(df, outpath, rundate, startPeriod, endPeriod, analytics, datasource):
	listOfCols0 = ['iso_timestamp','confidence_level', 'client_ip', 'client_port', 'User', 'Domain','server_name', 'SRC DNS/NETBIOS NAME', 'DST PORT','DST DNS/NETBIOS NAME', 'DST IP']
	listOfCols1 = df.columns
	
	for col in listOfCols0:
		if col not in listOfCols1:
			df = df.withColumn(col, F.lit("Null"))

	df1 = df.withColumnRenamed("client_ip", "SRC IP").withColumnRenamed('client_port', 'SRC PORT').withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('server_name', 'Server Name')

	df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

	df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

	df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')
	print "TRANSFORMING INFOBLOX DATAFRAME COMPLETED ------------------------------------->"

	return 0

def std_fireeye(df, outpath,rundate, startPeriod, endPeriod, analytics, datasource):
	listOfCols0 = ['iso_timestamp','confidence_level', 'client_ip', 'client_port', 'User', 'Domain', 'server_name', 'SRC DNS/NETBIOS NAME', 'DST PORT','DST DNS/NETBIOS NAME', 'DST IP']
	listOfCols1 = df.columns
	
	for col in listOfCols0:
		if col not in listOfCols1:
			df = df.withColumn(col, F.lit("Null"))

	df1 = df.withColumnRenamed("client_ip", "SRC IP").withColumnRenamed('client_port', 'SRC PORT').withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('server_name', 'Server Name')

	df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

	df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

	df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')
	print "TRANSFORMING INFOBLOX DATAFRAME COMPLETED ------------------------------------->"

	return 0

def std_bluecoat(df, outpath,rundate, startPeriod, endPeriod, analytics, datasource):
	listOfCols0 = ['iso_timestamp', 'confidence_level','server_name', 'c_ip', 'cs_host', 'cs_uri_port', 'cs_username', 'cs_auth_group', 'SRC DNS/NETBIOS NAME','DST IP', 'SRC PORT']
	# WHAT IS THE DOMAIN COL IN BLUECOAT
	listOfCols1 = df.columns
	
	for col in listOfCols0:
		if col not in listOfCols1:
			df = df.withColumn(col, F.lit("Null"))

	df1 = df.withColumnRenamed("c_ip", "SRC IP").withColumnRenamed('cs_uri_port', 'DST PORT').withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('cs_username', 'User').withColumnRenamed('cs_host', 'DST DNS/NETBIOS NAME').withColumnRenamed('cs_auth_group','Domain').withColumnRenamed('server_name', 'Server Name')

	df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

	df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

	df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')
	print "TRANSFORMING BLUECOAT DATAFRAME COMPLETED ------------------------------------->"

	return 0


def std_hipam(df, outpath, rundate, startPeriod, endPeriod, analytics, datasource):
	listOfCols0 = ['iso_timestamp','confidence_level', 'server_name', 'SRC IP', 'SRC PORT', 'Recipient ID', 'Domain', 'SRC DNS/NETBIOS NAME', 'DST PORT','DST DNS/NETBIOS NAME', 'DST IP']
	listOfCols1 = df.columns
	
	for col in listOfCols0:
		if col not in listOfCols1:
			df = df.withColumn(col, F.lit("Null"))

	df1 = df.withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('Recipient ID', 'User').withColumnRenamed('server_name', 'Server Name')

	df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

	df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

	df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')
	print "TRANSFORMING HIPAM DATAFRAME COMPLETED ------------------------------------->"

	return 0

def std_checkpoint(df, outpath,rundate, startPeriod, endPeriod, analytics, datasource):
	# FOR CONTROL LOGS
	if df.select('entry_type').limit(1).rdd.flatMap(lambda x: x).collect()[0] == 'control': # this is audit log
		listOfCols0 = ['iso_timestamp', 'confidence_level', 'client_ip', 'server_name', 'SRC PORT', 'uid', 'Domain', 'SRC DNS/NETBIOS NAME', 'DST PORT','DST DNS/NETBIOS NAME', 'DST IP']
		listOfCols1 = df.columns
	
		for col in listOfCols0:
			if col not in listOfCols1:
				df = df.withColumn(col, F.lit("Null"))

		df1 = df.withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('uid', 'User').withColumnRenamed('client_ip', 'SRC IP').withColumnRenamed('server_name', 'Server Name')

		df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

		df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

		df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True)
		print "TRANSFORMING CHECKPOINT DATAFRAME COMPLETED ------------------------------------->"

	else: # this is TRAFFIC log
		listOfCols0 = ['iso_timestamp', 'confidence_level', 'src_ip', 'server_name', 's_port', 'uid', 'Domain', 'SRC DNS/NETBIOS NAME', 'DST PORT','DST DNS/NETBIOS NAME', 'dst_ip']
		listOfCols1 = df.columns
	
		for col in listOfCols0:
			if col not in listOfCols1:
				df = df.withColumn(col, F.lit("Null"))

		df1 = df.withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('uid', 'User').withColumnRenamed('src_ip', 'SRC IP').withColumnRenamed('s_port', 'SRC PORT').withColumnRenamed('dst_ip', 'DST IP').withColumnRenamed('server_name', 'Server Name')

		df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

		df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain','Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')

		df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')
		print "TRANSFORMING CHECKPOINT DATAFRAME COMPLETED ------------------------------------->"

	return 0

def std_snare(df, outpath,rundate, startPeriod, endPeriod, analytics, datasource):
	listOfCols0 = ['iso_timestamp', 'confidence_level', 'server_name', 'src_ip', 's_port', 'username', 'domain', 'SRC DNS/NETBIOS NAME', 'DST PORT','DST DNS/NETBIOS NAME', 'DST IP']
	listOfCols1 = df.columns
	
	for col in listOfCols0:
		if col not in listOfCols1:
			df = df.withColumn(col, F.lit("Null"))

	df1 = df.withColumnRenamed("src_ip", "SRC IP").withColumnRenamed('s_port', 'SRC PORT').withColumnRenamed('username', 'User').withColumnRenamed('domain', 'Domain')\
		.withColumnRenamed('iso_timestamp', 'time of Event').withColumnRenamed('server_name', 'Server Name')

	df2 = df1.withColumn('Analytics', F.lit(analytics)).withColumn('Data Source', F.lit(datasource)).withColumn('Start Period', F.lit(startPeriod)).withColumn('End Period', F.lit(endPeriod))\
		.withColumn('Encoded Evidence', encode_udf('Evidence'))

	print "TRANSFORMING SNARE DATAFRAME COMPLETED ------------------------------------->"

	df3 = df2.select('Analytics','SRC IP','SRC PORT', 'SRC DNS/NETBIOS NAME', 'DST IP', 'DST PORT', 'DST DNS/NETBIOS NAME', 'User', 'Domain', 'Server Name', 'time of Event',\
	 	'Data Source', 'Start Period', 'End Period', 'confidence_level', 'Encoded Evidence')
	
	df3.repartition(1).write.csv(outpath +"/"+ rundate + "/" + analytics, header=True, mode = 'overwrite')

	return 0
