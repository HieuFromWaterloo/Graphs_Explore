############
#SOM PROJECT
#Requires minisom2.py
############

import pandas as pd
import numpy as np
import itertools
import statsmodels.formula.api as smf
import scipy.stats as scipystats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import concat,lower, col, lit, when, format_number,dayofmonth,hour,dayofyear,month,year,weekofyear,date_format
import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, ArrayType, FloatType, BooleanType, DoubleType
from pyspark.sql.types import StructField, StructType
import csv
from sklearn.preprocessing import StandardScaler
from minisom2 import MiniSom, fast_norm, bootstrap

conf_ = SparkConf().setAppName("som2")
sc = SparkContext(conf=conf_)
sqlContext = SQLContext(sc)
spark = SparkSession.builder.appName("integration").config(conf=conf_).getOrCreate() 


#inputFilePath = "/prod/18010/app/UO50/data/processed/elknas/bluecoataccess/event_date="
inputFilePath = "hdfs:/prod/18010/app/UO50/data/processed/elk_split/bluecoataccess/eventDate="
inputDate = '*'
outputFilePath = '/prod/18010/app/UO50/analytics/Alice/SOM/Attempt2/'

##########################################
#PREPARING DATA
##########################################

def readData(inputDate,category_str):
    path = inputFilePath + inputDate
    inputDf = sqlContext.read.parquet(path)\
                .where(lower(col('csCategories')).like("%{0}%".format(category_str)))
    reqDf = inputDf.withColumn('day', dayofyear(col('timeStamp')))\
                .select('csHost','cIP','timestamp','csUsername','scFilterResult','scBytes','csBytes','day')\
                .withColumn('Observed', when(lower(col('scFilterResult')) == 'observed', 1).otherwise(0))\
                .withColumn('User', when(col('csUsername').isNotNull(),col('csUsername') ).otherwise(col('cIP')))\
                .drop('cIP').drop('csUsername')
    return reqDf

def aggregateData(reqDf):
    agg_df = reqDf.groupby('csHost').agg(F.countDistinct('User').alias("num_of_users"),\
        F.count('timestamp').alias("num_of_Req"),\
        F.sum('scBytes'), F.sum('csBytes'), \
        F.countDistinct('day').alias('days_seen'), F.sum('Observed').alias('num_of_observed'))
    agg_df = agg_df.withColumn('total_bytes/req',\
            (agg_df['sum(scBytes)'] + agg_df['sum(csBytes)'])/agg_df['num_of_Req'])\
        .withColumn('diff_bytes/total_bytes', \
            (agg_df['sum(scBytes)'] -agg_df['sum(csBytes)'])/(agg_df['sum(scBytes)'] + agg_df['sum(csBytes)']))
    return agg_df

def scale_df_fn(pandasDf):
    scaler = StandardScaler()
    scaler.fit(pandasDf.iloc[:,1:])
    scaled_data = scaler.transform(pandasDf.iloc[:,1:])
    return scaled_data

def arrayToString(array):
    """converts array to comma separated string"""
    return '[' + ','.join([str(i) for i in array]) + ']'


##########################################
#SOM FINETUNE
##########################################
"""
To select the optimal combination, 
First in <function>FineTuneEntry, we calculate the goodness-of-fit measures of each model specified in the measures_lst field 
Then we calculate the ranking of the combinations based on each of the measures, with 1 being the combination with the most satisfactory value, and 18 being the least.
The combination with the smallest sum of rankings is then determined to be the most optimal
"""

def FineTuneEntry(scaled_data, x,y,sigma,learning_rate, num_iteration, measures_lst = ['qe', 'te'], Nboot = 0):
    """
    Mandatory Fields:
    x,y: the dimensions of som
    sigma: spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map
     (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
    learning_rate: initial learning rate
     (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
    num_iterations: number of times for SOM to randomly initialize weights

    measures_lst is a list of goodness-of-fit measures chosen to be calculated for the som model. List items can include
    	- 'qe'  (quantization error)
    	- 'te'  (topographic error)
    	- 'tf'  (topographic function)
    	- 'rqe' (relative_quant_error)
    	- 'rte'  (relative_topographic_error)
    	- 'qe_ci' (quantization error, confidence interval)
    	- 'te_ci'  (topographic error, confidence interval)
    
    Optional Field:
    Nboot: the number of bootstrap samples to generate
    (This field is specified to calculate measures rqe, rte, qe_ci, te_ci
     But this is often very computationally expensive - usually just keep it as default)
    """
    #train SOM
    som = MiniSom(x = x, y = y, input_len = scaled_data.shape[1], sigma = sigma, learning_rate = learning_rate)
    som.random_weights_init(scaled_data)
    som.train_random(data =scaled_data , num_iteration = num_iteration)  
    #calculate measures
    measure_dict = {'(x,y)': (x,y), 'sigma': sigma, 'learning_rate': learning_rate, 'num_iteration': num_iteration}
    if 'qe' in measures_lst: 
    	measure_dict['qe'] = som.quantization_error(scaled_data)
    if 'te' in measures_lst: 
    	measure_dict['te'] = som.topographic_error(scaled_data)
    if 'tf' in measures_lst:
    	top_func = [som.topographic_function(scaled_data, i) for i in [1,2,3,4,5,10,50]]
    	preserved = np.all(np.diff(top_func) <= 0) #returns True if seemingly monotone decreasing and converges to 0
    	measure_dict['tf'] = top_func[2]
    	measure_dict['preserved'] = preserved
    if any((True for x in ['rqe', 'rte', 'qe_ci', 'te_ci'] if x in measures_lst)):
    	qe_ci, te_ci, rqe, rte = bootstrap(Nboot, scaled_data, x,y, sigma, learning_rate, num_iteration)
    	measure_dict['Nboot'] = Nboot
    	measure_dict['qe_ci'], measure_dict['te_ci'],  = qe_ci, te_ci
    	measure_dict['rqe'], measure_dict['rte'] = rqe, rte
    return som, measure_dict


def select_finetune_ind(finetuneDf, measures_lst):
	"""
	returns the index of optimal (hyperparameter combination) record in finetuneDf
	"""
	score = pd.Series([0]*finetuneDf.shape[0])
	measureSum = pd.Series([0]*finetuneDf.shape[0])
	for col in list(set(['qe', 'te', 'rqe', 'rte']) & set(measures_lst)):
		score += finetuneDf[col].rank(ascending = True)
		measureSum += finetuneDf[col]
	if 'tf' in measures_lst:
		score += finetuneDf['tf'].apply(lambda x: abs(x)).rank(ascending = True)
		measureSum += finetuneDf['tf'].apply(lambda x: abs(x))
		score += finetuneDf['preserved'].rank(ascending = False) * 2  #give higher weight, because important
		measureSum += finetuneDf['preserved']
	for col in list(set(['qe_ci', 'te_ci']) & set(measures_lst)):
		low, high = finetuneDf[col].apply(lambda x: x[0]), finetuneDf[col].apply(lambda x: x[1])
		score += (high - low).rank(ascending = True)  
		measureSum += low + high
	score += measureSum.rank(ascending = True)
	return score.idxmin()


def SOM_Selection(scaled_data, measures_lst, param_combos):
	#for each of hyper-parameter combinations,
	# the corresponding fitted model is saved in the list ListOfSOM_model, the goodness-of-fit of the model is saved as a dictionary in ListOfmeasure_dict
	ListOfSOM_model = []
	ListOfmeasure_dict = []
	for param_dict in param_combos:
		som, measure_dict = FineTuneEntry(scaled_data, x=param_dict['x'],y=param_dict['y'],sigma=param_dict['sigma'],\
			learning_rate = param_dict['learning_rate'], num_iteration=param_dict['num_iteration'],\
			measures_lst = measures_lst, Nboot = param_dict['Nboot'])
		ListOfmeasure_dict.append(measure_dict)
		ListOfSOM_model.append(som)
	best_ind = select_finetune_ind(pd.DataFrame(ListOfmeasure_dict), measures_lst)
	arraytoString_udf = F.udf(arrayToString, StringType())
	FinetuneSummary = sqlContext.createDataFrame(pd.DataFrame(ListOfmeasure_dict))\
						.withColumn('(x,y)', arraytoString_udf('(x,y)'))
	FinetuneSummary.write.option('header', 'true').csv(outputFilePath + 'FinetuneSummary')
	print('FinetuneSummary is saved!')
	return ListOfmeasure_dict[best_ind], ListOfSOM_model[best_ind]



###############################################
#REGRESSION ANALYSIS (noSOM)- METHOD 0
################################################

def reg_outliers_std(df, x, y):
	"""outputs df with corresponding regression outliers"""
	x1, y1 = df[x], df[y]
	lm = sm.OLS(y1, sm.add_constant(x1)).fit()
	sigma_err = np.sqrt(lm.scale)
	scaled_resid = (lm.resid - lm.resid.mean())/ (sigma_err)
	return df[(scaled_resid > 3) | (scaled_resid < -3)]

def plot_outliers(df, x, y):
	x1, y1 = df[x], df[y]
	lm = sm.OLS(y1, sm.add_constant(x1)).fit()
	sigma_err = np.sqrt(lm.scale)
	scaled_resid = (lm.resid - lm.resid.mean())/ (sigma_err)
	plt.scatter(np.sort(x1), y1[np.argsort(x1)])
	plt.plot(np.sort(x1), lm.predict()[np.argsort(x1)], label = "regression")
	plt.scatter(input[(scaled_resid > 3) | (scaled_resid < -3)]['num_of_Req']\
            , input[(scaled_resid > 3) | (scaled_resid < -3)]['sum(csBytes)'], color = "green")

def reg_influential_points(df, x, y): 
	"""outputs df with corresponding to leverage points"""
	x1, y1 = df[x], df[y]
	lm = sm.OLS(y1, sm.add_constant(x1)).fit()
	p = 2
	n = df.shape[0]
	infl = lm.get_influence()
	leviers = infl.hat_matrix_diag
	threshold = 2*float(p+1)/n
	return df[leviers > threshold]

def sublist(lst1, lst2):
	"""determines if a list is a sublist of the other"""
	ls1 = [element for element in lst1 if element in lst2]
	ls2 = [element for element in lst2 if element in lst1]
	return ls1 == ls2

def regress_outliers(df,lst_of_xy):
	"""
	lst_of_xy is a list of tuples (x,y) which correspond to regression column names
	Outputs outlier website names based on regression outliers and leverage points
	"""
	websites = []
	for (x,y) in lst_of_xy:
		lst1 = reg_outliers_std(df,x,y)['csHost'].tolist()
		lst2 = reg_influential_points(df,x,y)['csHost'].tolist()
		if len(lst1) <= len(lst2) and sublist(lst1, lst2):
			websites = list(set(websites).union(lst1))
		else:
			websites = list(set(websites).union(lst2))
	return websites

def StatAnomalies(df, lst_of_xy):
	"""	Returns pandas dataframe of anomalies detected through statistical regression analysis"""
	regWebsites = regress_outliers(pandasDf, lst_of_xy)
	return df[df['csHost'].isin(regWebsites)]



#######################################################
#ANOMALY SELECTIONS - METHOD 1: U-Matrix (ABANDONED)
#######################################################
"""
*ABANDONED*
This approach is equivalent to determining the "whiter" neurons in the SOM map visualization done in minisom2. 
The unified distance matrix(U-matrix) of a neuron is the normalized sum of distances between the neuron and its neighbors. 
The larger the U-matrix value of a neuron is, the greater the distance between the neuron and its neighbors.
We calculate the U-matrix values for all neurons, look for the outliers in its distribution, and determine website data points mapped to the outlier neurons to be anomalies. 
	When searching outliers in the distribution, we first attempt to transform the distance distribution into a normal distribution using <function>normal_transformation. 
	The transformation is determined to be successful if the transformed data passes the Jarque-Bera test. 
	If successful, outlier data points will be identified as those whose transformed distance measures are larger than the mean + 3 standard deviations of the normal distribution. 
	If the transformation is not successful, a threshold is selected to 0.9 or 0.1 based on distribution skewness.
This method is not soundly proven and gave extremely fluctuating results; thus was abandoned due to low performance and unreliability.
"""

def normal_transformation(lst):
	"""
	This function attempts normalizing transformations on lst
	Among transformations that pass the Jarque Bera test (do not reject normality), select the transformation with the largest p-value (or smallest JBstat) 
	If none of the transformations pass the JB test, no transformation will be performed 
	Resulting descr can be one of the following: 'anscrombe', 'boxcox with lambda ' + lambdaVal, 'No transformation'
	"""
	transfDict = {}
	lst = np.array(lst)
	if np.sum(lst < 0) == 0 and stats.skewtest(lst)[1] < 0.05 and stats.skew(lst) > 0:
		#anscrombe transform
		try:
			transformation = map(lambda x: 2*math.sqrt(x+3/8), lst)
			JBstat, JBpval = stats.jarque_bera(transformation)
			if JBpval > 0.05: transfDict[JBpval] = (transformation, 'anscrombe')
		except:
			pass
	#box-cox 
	try: 
		transformation, lambdaVal = stats.boxcox(lst)
		JBstat, JBpval = stats.jarque_bera(transformation)
		if JBpval > 0.05: transfDict[JBpval] = (transformation, 'boxcox with lambda ' + str(lambdaVal))
	except:
		pass
	if len(transfDict) == 0: 
		best_transformation, descr = lst, 'No transformation'
	else:
		best_transformation, descr = transfDict[max(transfDict.keys())]
	return best_transformation, descr


def visualClusterSelection(som_model):
	"""	returns list of coordinates of outlier neurons 	"""
	DistMap = som_model.distance_map()
	transformation, descr = normal_transformation(DistMap.ravel())
	print('The following transformation is performed: ' + descr)
	skewVal = stats.skew(DistMap.ravel())
	if descr == 'No transformation':
		if skewVal > 0: 
			return zip(np.where(DistMap > 0.9)[0], np.where(DistMap > 0.9)[1])
		else:
			return zip(np.where(DistMap < 0.1)[0], np.where(DistMap < 0.1)[1])
	elif 'anscrombe' in descr:
		iqr = np.subtract(*np.percentile(transformation, [75, 25]))
		low, high = map(lambda x: (x/2.)**2. - 3./8., \
			[np.mean(transformation) - 1.5*iqr, np.mean(transformation) + 1.5*iqr])
		return zip(np.where(DistMap > high)[0], np.where(DistMap > high)[1])	
	elif 'boxcox' in descr:
		iqr = np.subtract(*np.percentile(transformation, [75, 25]))
		lbda = float(descr[19:])
		#print(lbda)
		iqr = np.subtract(*np.percentile(transformation, [75, 25]))
		low, high = inv_boxcox([np.mean(transformation) - 1.5*iqr, np.mean(transformation) + 1.5*iqr], lbda)
		low, high = sorted([low, high]) #need to sort b/c if lbda negative, low > high
		if skewVal >= 0: 
			return zip(np.where(DistMap > high)[0], np.where(DistMap > high)[1])
		else:
			return zip(np.where(DistMap < low)[0], np.where(DistMap < low)[1])
	return []


def VisualAnomalies(somModel, scaled_data):
	"""Returns pandas dataframe of anomalies detected through visual method"""
	#Finds list of coordinates of outlier neurons
	coord_lst = visualClusterSelection(somModel)
	wm = somModel.win_map(scaled_data)
	print(coord_lst)
	#filters the outlier neurons so that only neurons with less than 100 data points mapped to are investigated
	filteredLst = filter(lambda item: len(wm[item])< 100 and len(wm[item]) > 0, coord_lst)
	print('Selected Anomalous clusters with coordinates ' + str(filteredLst))
	if np.sum([len(wm[item]) for item in filteredLst]) > 0:
		entries = np.concatenate([wm[item] for item in filteredLst], axis = 0)
		idxLst = [np.where((scaled_data == row).all(axis = 1))[0][0] for row in entries]
		print(idxLst)
		return pandasDf.iloc[idxLst, ]
	return pd.DataFrame(columns = pandasDf.columns)


##########################################################################
#ANOMALY SELECTIONS - METHOD 2: BACK TRACKING WITH TRUE POSITIVES(ABANDONED)
##########################################################################

"""
*ABANDONED*
We attempt to determine anomalies in the data by identifying the websites that are mapped to the same neurons as the websites ("knownWebsites")
 	- found by the REGRESSION ANALYSIS PORTION (because results regularly exhibit high statistical significance)
 	- determined via visualization
 	- known true positives identified by SME. Unfortunately,  findings through this approach account for about 5% of the full data set, which is too high. 
Unfortunately,  findings through this approach account for about 5% of the full data set, which is too high
"""

def BackTrackAnomalies(somModel, scaled_data, knownWebsites):
	"""
	Returns pandas dataframe of anomalies detected through back tracking method
	"""
	knownEntries = scaled_data[pandasDf['csHost'].isin(knownWebsites)]
	coord_lst_dup = map(lambda ls: tuple(ls), \
					list(np.apply_along_axis(lambda x: somModel.winner(x), 1, knownEntries)))
	coord_lst = list(set(coord_lst_dup))
	print(str(len(coord_lst_dup)) + " known anomalies fall under " + str(len(coord_lst)) + " clusters ")
	wm = somModel.win_map(scaled_data)
	filteredLst = filter(lambda item: len(wm[item])< 100 and len(wm[item]) > 0, coord_lst)
	entries = np.concatenate([wm[item] for item in filteredLst], axis = 0)
	idxLst = [np.where((scaled_data == row).all(axis = 1))[0][0] for row in entries]
	return pandasDf.iloc[idxLst, ]


#################################################
#ANOMALY SELECTIONS - METHOD 3: GLOBAL DETECTION
#################################################

"""
For each of the input vectors, one can quantify its distance to the mapped two-dimensional lattice by calculating the Euclidean distance between the input vector and its BMU’s weights. 
We hereby call this measure the “distance-to-map” of an input vector. 
If the SOM model is appropriately chosen, the majority of these distance-to-map values should be relatively small, and the input vectors corresponding to large distance-to-map values will likely indicate an irregularity.
 As a result, after collecting the distance-to-map measures of all the input vectors, one can perform outlier analysis on the right tail of the distance distribution to determine possible anomalies. 
 However, outlier threshold selection often relies on graphing using <function>plotEntryErrors (or SME input) 
"""

def plotEntryErrors(somModel, scaled_data):
	entryErrorDct = somModel.distance_dict(scaled_data)
	errors = np.array(map(lambda tpl: tpl[1], entryErrorDct))
	print(stats.describe(errors))
	#plt.figure()
	#sns.distplot(errors)
	#plt.figure()
	#sns.distplot(errors[errors > max(errors)/4])
	print('There are {0} errors larger than {1}'.format(errors[errors > max(errors)/10].shape[0],  max(errors)/10))
	print('There are {0} errors larger than 10'.format(errors[errors >= 10].shape[0]))
	print('There are {0} errors larger than 20'.format(errors[errors >= 20].shape[0]))
	print('There are {0} errors larger than 50'.format(errors[errors >= 50].shape[0]))


def GlobalAnomalies(somModel, scaled_data, threshold):
	entryErrorDct = somModel.distance_dict(scaled_data)
	filteredDct = filter(lambda tpl: tpl[1] > threshold, entryErrorDct)
	entries = np.array(map(lambda tpl: tpl[0], filteredDct))
	idxLst = [np.where((scaled_data == row).all(axis = 1))[0][0] for row in entries]
	return pandasDf.iloc[idxLst, ]


#################################################
#SAVING DATAFRAMES
#################################################
def saveAllDf(outDfname, *dfs):
	unionDf = sqlContext.createDataFrame(sc.emptyRDD(), StructType([]))
	for df in dfs:
		if df.shape[0] > 0:
			sparkDf = sqlContext.createDataFrame(df)
			try:
				sparkDf.write.csv(outputFilePath + df.name)
				print('Saved DataFrame ' + df.name)
			except:
				pass
			if df.shape[0] < scaled_data.shape[0]*0.01:
				if unionDf.rdd.isEmpty():
					unionDf = sparkDf
				else:
					unionDf = unionDf.union(sparkDf)
	unionDf = unionDf.drop_duplicates()
	unionDf.write.csv(outputFilePath + outDfname)
	print('Saved DataFrame ' + outDfname)
	return unionDf


#regression analysis is performed on the following feature combinations
lst_of_xy = [('num_of_observed', 'num_of_Req'), ('num_of_observed', 'sum(csBytes)'), ('num_of_Req', 'sum(csBytes)')]
#Fields needed for BackTrackAnomalies (commented out because approach abandoned)
###HARDCODED anomalies obtained by visualation:
###observedWebsites = ['cdn.gty.org', 'www.abidingradio.org', 'switchcreative.ca', 'g.christianbook.com', 'www.blueletterbible.org', 'storage.cloversites.com', 'www1.cbn.com', 'www.theshirdisaimandir.ca']
###regWebsites = regress_outliers(pandasDf, lst_of_xy)
###TPWebsites = list(set().union(regWebsites, observedWebsites))

if __name__ == "__main__":
    reqDf = readData(inputDate, 'relig')
    aggDf = aggregateData(reqDf)
    pandasDf = aggDf.toPandas()
    scaled_data = scale_df_fn(pandasDf)

    measures_lst = ['qe', 'te']
	param_combos = [{'x': 10,'y': 10,'sigma': 4,'learning_rate': 0.5, 'num_iteration': 1000, 'Nboot': 0}, \
					{'x': 20,'y': 20,'sigma': 8,'learning_rate': 0.5, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 20,'y': 20,'sigma': 8,'learning_rate': 1, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 27,'y': 27,'sigma': 13,'learning_rate': 1, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 27,'y': 27,'sigma': 12,'learning_rate': 1, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 27,'y': 27,'sigma': 10,'learning_rate': 1, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 27,'y': 27,'sigma': 12,'learning_rate': 0.5, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 27,'y': 27,'sigma': 12,'learning_rate': 0.5, 'num_iteration': 2000, 'Nboot': 0},\
					{'x': 27,'y': 27,'sigma': 12,'learning_rate': 0.5, 'num_iteration': 500, 'Nboot': 0},\
					{'x': 28,'y': 28,'sigma': 14,'learning_rate': 0.5, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 28,'y': 28,'sigma': 12,'learning_rate': 0.5, 'num_iteration': 1000, 'Nboot': 0},\
					{'x': 28,'y': 28,'sigma': 10,'learning_rate': 0.5, 'num_iteration': 1000, 'Nboot': 0}\
					]

    best_measureDict, best_som = SOM_Selection(scaled_data, measures_lst, param_combos)
    statDf = StatAnomalies(pandasDf, lst_of_xy)
    #visualDf = VisualAnomalies(best_som, scaled_data)
   	#backtrackDf = BackTrackAnomalies(best_som, scaled_data, TPWebsites)
   	#plotEntryErrors(best_som, scaled_data)
   	globalDf = GlobalAnomalies(best_som, scaled_data, 10)
   	statDf.name, globalDf.name = 'statDf','globalDf'
   	finalDf = saveAllDf('finalDf', statDf, globalDf)