# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:02:22 2024

@author: UOBASUB
"""


import csv
import io
from datetime import datetime, timedelta, date, time
import pandas as pd 
import numpy as np
import math
from google.cloud.storage import Blob
from google.cloud import storage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None
from scipy.signal import argrelextrema
from scipy import signal
from scipy.misc import derivative
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import plotly.io as pio
pio.renderers.default='browser'
import bisect

def ema(df):
    df['30ema'] = df['close'].ewm(span=30, adjust=False).mean()
    df['40ema'] = df['close'].ewm(span=40, adjust=False).mean()
    df['28ema'] = df['close'].ewm(span=28, adjust=False).mean()
    df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
    df['100ema'] = df['close'].ewm(span=100, adjust=False).mean()
    df['150ema'] = df['close'].ewm(span=150, adjust=False).mean()
    df['200ema'] = df['close'].ewm(span=200, adjust=False).mean()
    df['1ema'] = df['close'].ewm(span=1, adjust=False).mean()
    df['3ema'] = df['close'].ewm(span=3, adjust=False).mean()
    df['5ema'] = df['close'].ewm(span=5, adjust=False).mean()
    df['6ema'] = df['close'].ewm(span=6, adjust=False).mean()
    df['8ema'] = df['close'].ewm(span=8, adjust=False).mean()
    df['12ema'] = df['close'].ewm(span=12, adjust=False).mean()
    df['15ema'] = df['close'].ewm(span=15, adjust=False).mean()
    df['20ema'] = df['close'].ewm(span=20, adjust=False).mean()


def Bbands(df):
    df['BbandsMid'] = df['close'].rolling(20).mean()
    df['BbandsUpp'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
    df['BbandsLow'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)

    #return [df['BbandsUpp'][len(df['BbandsUpp'])-1], df['BbandsLow'][len(df['BbandsLow'])-1]]


def vwap(df):
    v = df['volume'].values
    h = df['high'].values
    l = df['low'].values
    # print(v)
    df['vwap'] = np.cumsum(v*(h+l)/2) / np.cumsum(v)
    #df['disVWAP'] = (abs(df['close'] - df['vwap']) / ((df['close'] + df['vwap']) / 2)) * 100
    #df['disVWAPOpen'] = (abs(df['open'] - df['vwap']) / ((df['open'] + df['vwap']) / 2)) * 100
    #df['disEMAtoVWAP'] = ((df['close'].ewm(span=12, adjust=False).mean() - df['vwap'])/df['vwap']) * 100

    df['volumeSum'] = df['volume'].cumsum()
    df['volume2Sum'] = (v*((h+l)/2)*((h+l)/2)).cumsum()
    #df['myvwap'] = df['volume2Sum'] / df['volumeSum'] - df['vwap'].values * df['vwap']
    #tp = (df['low'] + df['close'] + df['high']).div(3).values
    # return df.assign(vwap=(tp * v).cumsum() / v.cumsum())


def sigma(df):
    try:
        val = df.volume2Sum / df.volumeSum - df.vwap * df.vwap
    except(ZeroDivisionError):
        val = df.volume2Sum / (df.volumeSum+0.000000000001) - df.vwap * df.vwap
    return math.sqrt(val) if val >= 0 else val


def PPP(df):

    df['STDEV_TV'] = df.apply(sigma, axis=1)
    stdev_multiple_0 = 0.50
    stdev_multiple_1 = 1
    stdev_multiple_1_5 = 1.5
    stdev_multiple_2 = 2.00
    stdev_multiple_25 = 2.50

    df['STDEV_0'] = df.vwap + stdev_multiple_0 * df['STDEV_TV']
    df['STDEV_N0'] = df.vwap - stdev_multiple_0 * df['STDEV_TV']

    df['STDEV_1'] = df.vwap + stdev_multiple_1 * df['STDEV_TV']
    df['STDEV_N1'] = df.vwap - stdev_multiple_1 * df['STDEV_TV']
    
    df['STDEV_15'] = df.vwap + stdev_multiple_1_5 * df['STDEV_TV']
    df['STDEV_N15'] = df.vwap - stdev_multiple_1_5 * df['STDEV_TV']

    df['STDEV_2'] = df.vwap + stdev_multiple_2 * df['STDEV_TV']
    df['STDEV_N2'] = df.vwap - stdev_multiple_2 * df['STDEV_TV']
    
    df['STDEV_25'] = df.vwap + stdev_multiple_25 * df['STDEV_TV']
    df['STDEV_N25'] = df.vwap - stdev_multiple_25 * df['STDEV_TV']


def VMA(df):
    df['vma'] = df['volume'].rolling(4).mean()
      

def historV1(df, num, quodict, trad:list=[], quot:list=[], rangt:int=1):
    #trad = AllTrades
    pzie = [(i[0],i[1]) for i in trad if i[1] >= rangt]
    dct ={}
    for i in pzie:
        if i[0] not in dct:
            dct[i[0]] =  i[1]
        else:
            dct[i[0]] +=  i[1]
            
    
    pzie = [i for i in dct ]#  > 500 list(set(pzie))
    
    hist, bin_edges = np.histogram(pzie, bins=num)
    
    cptemp = []
    zipList = []
    cntt = 0
    for i in range(len(hist)):
        pziCount = 0
        acount = 0
        bcount = 0
        ncount = 0
        for x in trad:
            if bin_edges[i] <= x[0] < bin_edges[i+1]:
                pziCount += (x[1])
                if x[4] == 'A':
                    acount += (x[1])
                elif x[4] == 'B':
                    bcount += (x[1])
                elif x[4] == 'N':
                    ncount += (x[1])
                
        #if pziCount > 100:
        cptemp.append([bin_edges[i],pziCount,cntt,bin_edges[i+1]])
        zipList.append([acount,bcount,ncount])
        cntt+=1
        
    for i in cptemp:
        i+=countCandle(trad,[],i[0],i[3],df['name'][0],{})

    for i in range(len(cptemp)):
        cptemp[i] += zipList[i]
        
    
    sortadlist = sorted(cptemp, key=lambda stock: float(stock[1]), reverse=True)
    
    return [cptemp,sortadlist] 


def historV2(df, num, quodict, trad:list=[], quot:list=[], rangt:int=1):
    #trad = AllTrades
    pzie = [(i[0],i[1]) for i in trad if i[1] >= rangt]
    dct ={}
    for i in pzie:
        if i[0] not in dct:
            dct[i[0]] =  i[1]
        else:
            dct[i[0]] +=  i[1]
            
    
    pzie = [i for i in dct ]#  > 500 list(set(pzie))
    
    hist, bin_edges = np.histogram(pzie, bins=num)
    
    cptemp = []
    zipList = []
    cntt = 0
    for i in range(len(hist)):
        pziCount = 0
        acount = 0
        bcount = 0
        ncount = 0
        for x in trad:
            if bin_edges[i] <= x[0] < bin_edges[i+1]:
                pziCount += (x[1])
                if x[5] == 'A':
                    acount += (x[1])
                elif x[5] == 'B':
                    bcount += (x[1])
                elif x[5] == 'N':
                    ncount += (x[1])
                
        #if pziCount > 100:
        cptemp.append([bin_edges[i],pziCount,cntt,bin_edges[i+1]])
        zipList.append([acount,bcount,ncount])
        cntt+=1
        
    for i in cptemp:
        i+=countCandle(trad,[],i[0],i[3],df['name'][0],{})

    for i in range(len(cptemp)):
        cptemp[i] += zipList[i]
        
    
    sortadlist = sorted(cptemp, key=lambda stock: float(stock[1]), reverse=True)
    
    return [cptemp,sortadlist] 


def countCandle(trad,quot,num1,num2, stkName, quodict):
    enum = ['Bid(SELL)','BelowBid(SELL)','Ask(BUY)','AboveAsk(BUY)','Between']
    color = ['red','darkRed','green','darkGreen','black']

   
    lsr = splitHun(stkName,trad, quot, num1, num2, quodict)
    ind = lsr.index(max(lsr))   #lsr[:4]
    return [enum[ind],color[ind],lsr]


def splitHun(stkName, trad, quot, num1, num2, quodict):
    Bidd = 0
    belowBid = 0
    Askk = 0
    aboveAsk = 0
    Between = 1
    
    return [Bidd,belowBid,Askk,aboveAsk,Between]
 

def valueAreaV1(lst):
    mkk = [i for i in lst if i[1] > 0]
    if len(mkk) == 0:
        mkk = hs[0]
    for xm in range(len(mkk)):
        mkk[xm][2] = xm
        
    pocIndex = sorted(mkk, key=lambda stock: float(stock[1]), reverse=True)[0][2]
    sPercent = sum([i[1] for i in mkk]) * .70
    pocVolume = mkk[mkk[pocIndex][2]][1]
    #topIndex = pocIndex - 2
    #dwnIndex = pocIndex + 2
    topVol = 0
    dwnVol = 0
    total = pocVolume
    #topBool1 = topBool2 = dwnBool1 = dwnBool2 =True

    if 0 <= pocIndex - 1 and 0 <= pocIndex - 2:
        topVol = mkk[mkk[pocIndex - 1][2]][1] + mkk[mkk[pocIndex - 2][2]][1]
        topIndex = pocIndex - 2
        #topBool2 = True
    elif 0 <= pocIndex - 1 and 0 > pocIndex - 2:
        topVol = mkk[mkk[pocIndex - 1][2]][1]
        topIndex = pocIndex - 1
        #topBool1 = True
    else:
        topVol = 0
        topIndex = pocIndex

    if pocIndex + 1 < len(mkk) and pocIndex + 2 < len(mkk):
        dwnVol = mkk[mkk[pocIndex + 1][2]][1] + mkk[mkk[pocIndex + 2][2]][1]
        dwnIndex = pocIndex + 2
        #dwnBool2 = True
    elif pocIndex + 1 < len(mkk) and pocIndex + 2 >= len(mkk):
        dwnVol = mkk[mkk[pocIndex + 1][2]][1]
        dwnIndex = pocIndex + 1
        #dwnBool1 = True
    else:
        dwnVol = 0
        dwnIndex = pocIndex

    # print(pocIndex,topVol,dwnVol,topIndex,dwnIndex)
    while sPercent > total:
        if topVol > dwnVol:
            total += topVol
            if total > sPercent:
                break

            if 0 <= topIndex - 1 and 0 <= topIndex - 2:
                topVol = mkk[mkk[topIndex - 1][2]][1] + \
                    mkk[mkk[topIndex - 2][2]][1]
                topIndex = topIndex - 2

            elif 0 <= topIndex - 1 and 0 > topIndex - 2:
                topVol = mkk[mkk[topIndex - 1][2]][1]
                topIndex = topIndex - 1

            if topIndex == 0:
                topVol = 0

        else:
            total += dwnVol

            if total > sPercent:
                break

            if dwnIndex + 1 < len(mkk) and dwnIndex + 2 < len(mkk):
                dwnVol = mkk[mkk[dwnIndex + 1][2]][1] + \
                    mkk[mkk[dwnIndex + 2][2]][1]
                dwnIndex = dwnIndex + 2

            elif dwnIndex + 1 < len(mkk) and dwnIndex + 2 >= len(mkk):
                dwnVol = mkk[mkk[dwnIndex + 1][2]][1]
                dwnIndex = dwnIndex + 1

            if dwnIndex == len(mkk)-1:
                dwnVol = 0

        if dwnIndex == len(mkk)-1 and topIndex == 0:
            break
        elif topIndex == 0:
            topVol = 0
        elif dwnIndex == len(mkk)-1:
            dwnVol = 0

        # print(total,sPercent,topIndex,dwnIndex,topVol,dwnVol)
        # time.sleep(3)

    return [mkk[topIndex][0], mkk[dwnIndex][0], mkk[pocIndex][0]]



def find_clusters(numbers, threshold):
    clusters = []
    current_cluster = [numbers[0]]

    # Iterate through the numbers
    for i in range(1, len(numbers)):
        # Check if the current number is within the threshold distance from the last number in the cluster
        if abs(numbers[i] - current_cluster[-1]) <= threshold:
            current_cluster.append(numbers[i])
        else:
            # If the current number is outside the threshold, store the current cluster and start a new one
            clusters.append(current_cluster)
            current_cluster = [numbers[i]]

    # Append the last cluster
    clusters.append(current_cluster)
    
    return clusters

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

symbolNumList = ['118', '4358', '42012334', '585200', '2017','28080', '938']
symbolNameList = ['ES', 'NQ', 'YM','CL', 'GC', 'HG', 'NG']


gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage")

from dash import Dash, dcc, html, Input, Output, callback, State
inter = 100000#60000
app = Dash()
app.layout = html.Div([
    
    dcc.Graph(id='graph'),
    dcc.Interval(
        id='interval',
        interval=inter,
        n_intervals=0,
      ),

    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',children="Enter a symbol from |'ES', 'NQ', 'YM','CL', 'GC', 'HG', 'NG'| and submit"),
    dcc.Store(id='stkName-value')
])

@callback(
    Output('stkName-value', 'data'),
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value'),
    prevent_initial_call=True
)

def update_output(n_clicks, value):
    value = str(value).upper().strip()
    
    if value in symbolNameList:
        print('The input symbol was "{}" '.format(value))
        return str(value).upper(), str(value).upper()
    
    
    else:
        return 'The input symbol was '+str(value)+" is not accepted please try different symbol from  |'ES', 'NQ', 'YM','CL', 'GC', 'HG', 'NG'|  ", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(Output('graph', 'figure'),
          Input('interval', 'n_intervals'),
          State('stkName-value', 'data'))


    


def update_graph_live(n_intervals, data):
    print('inFunction')	

    if data in symbolNameList:
        stkName = data
        symbolNum = symbolNumList[symbolNameList.index(stkName)] 
    else:
        stkName = 'NQ'  
        symbolNum = symbolNumList[symbolNameList.index(stkName)]


    
    interval = '20'
    
    blob = Blob('FuturesOHLC'+str(symbolNum), bucket) 
    FuturesOHLC = blob.download_as_text()
        
    
    csv_reader  = csv.reader(io.StringIO(FuturesOHLC))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    
    aggs = [ ]  
    newOHLC = [i for i in csv_rows]
    
    for i in newOHLC:
        hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
        if hourss < 10:
            hourss = '0'+str(hourss)
        minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
        if minss < 10:
            minss = '0'+str(minss)
        opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
        aggs.append([int(i[2])/1e9, int(i[3])/1e9, int(i[4])/1e9, int(i[5])/1e9, int(i[6]), opttimeStamp, int(i[0]), int(i[1])])
        
            
    newAggs = []
    for i in aggs:
        if i not in newAggs:
            newAggs.append(i)
            
          
    df = pd.DataFrame(newAggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',])
    
    df['strTime'] = df['timestamp'].apply(lambda x: pd.Timestamp(int(x) // 10**9, unit='s', tz='EST') )
    
    df.set_index('strTime', inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    df_resampled = df.resample(interval+'T').agg({
        'timestamp': 'first',
        'name': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'first',
        'volume': 'sum'
    })
    
    df_resampled.reset_index(drop=True, inplace=True)
    
    df = df_resampled
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True) 
    
    vwap(df)
    ema(df)
    PPP(df)
    #Bbands(df)
    
    
    blob = Blob('FuturesTrades'+str(symbolNum), bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    #STrades = [i for i in csv_rows]
    AllTrades = []
    for i in csv_rows:
        hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
        if hourss < 10:
            hourss = '0'+str(hourss)
        minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
        if minss < 10:
            minss = '0'+str(minss)
        opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
        AllTrades.append([int(i[1])/1e9, int(i[2]), int(i[0]), 0, i[3], opttimeStamp])
        
    #AllTrades = [i for i in AllTrades if i[1] > 2]        
            
    hs = historV1(df,50,{},AllTrades,[])
    
    va = valueAreaV1(hs[0])
    
    x = np.array([i for i in range(len(df))])
    y = np.array([i for i in df['40ema']])
    
    
    
    # Simple interpolation of x and y
    f = interp1d(x, y)
    x_fake = np.arange(0.1, len(df)-1, 1)
    
    # derivative of y with respect to x
    df_dx = derivative(f, x_fake, dx=1e-6)
    df_dx = np.append(df_dx, df_dx[len(df_dx)-1])
    
    mTrade = [i for i in AllTrades]
    
     
    mTrade = sorted(mTrade, key=lambda d: d[1], reverse=True)
    
    [mTrade[i].insert(4,i) for i in range(len(mTrade))] 
    
    newwT = []
    for i in mTrade:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
    
    
    dtime = df['time'].dropna().values.tolist()
    dtimeEpoch = df['timestamp'].dropna().values.tolist()
    
    
    tempTrades = [i for i in AllTrades]
    tempTrades = sorted(tempTrades, key=lambda d: d[6], reverse=False) 
    tradeTimes = [i[6] for i in tempTrades]
    
    timeDict = {}
    for ttm in dtime:
        for tradMade in tempTrades[bisect.bisect_left(tradeTimes, ttm):]:
            if datetime.strptime(tradMade[6], "%H:%M:%S") > datetime.strptime(ttm, "%H:%M:%S") + timedelta(minutes=int(interval)):
                try:
                    timeDict[ttm] += [timeDict[ttm][0]/sum(timeDict[ttm]), timeDict[ttm][1]/sum(timeDict[ttm]), timeDict[ttm][2]/sum(timeDict[ttm])]
                except(KeyError,ZeroDivisionError):
                    timeDict[ttm] = [0,0,0]
                break
            
            if ttm not in timeDict:
                timeDict[ttm] = [0,0,0]
            if ttm in timeDict:
                if tradMade[5] == 'B':
                    timeDict[ttm][0] += tradMade[1]#tradMade[0] * tradMade[1]
                elif tradMade[5] == 'A':
                    timeDict[ttm][1] += tradMade[1]#tradMade[0] * tradMade[1] 
                elif tradMade[5] == 'N':
                    timeDict[ttm][2] += tradMade[1]#tradMade[0] * tradMade[1] 
                
    
    for i in timeDict:
        if len(timeDict[i]) == 3:
            try:
                timeDict[i] += [timeDict[i][0]/sum(timeDict[i]), timeDict[i][1]/sum(timeDict[i]), timeDict[i][2]/sum(timeDict[i])]
            except(ZeroDivisionError,KeyError):
                timeDict[i] += [0,0,0]
    
    
    timeFrame = [[i,'']+timeDict[i] for i in timeDict]
    
    for i in range(len(timeFrame)):
        timeFrame[i].append(dtimeEpoch[i])
        
    
    #df['superTrend'] = ta.supertrend(df['high'], df['low'], df['close'], length=2, multiplier=1.5)['SUPERTd_2_1.5']
    #df['superTrend'][df['superTrend'] < 0] = 0
    
    blob = Blob('PrevDay', bucket) 
    PrevDay = blob.download_as_text()
        
    
    csv_reader  = csv.reader(io.StringIO(PrevDay))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    previousDay = [csv_rows[[i[4] for i in csv_rows].index(symbolNum)][0] ,csv_rows[[i[4] for i in csv_rows].index(symbolNum)][1] ,csv_rows[[i[4] for i in csv_rows].index(symbolNum)][2]]
    
    '''
    ----------------------------------------------------------------
    '''
    
    df['prevDayLVA'] = [float(previousDay[0])]*len(df['time'])
    df['prevDayHVA'] = [float(previousDay[1])]*len(df['time'])
    df['prevDayPOC'] = [float(previousDay[2])]*len(df['time'])
    
    df['buyCount'] = pd.Series([i[2] for i in timeFrame])
    df['sellCount'] = pd.Series([i[3] for i in timeFrame])
    
    df['buyDecimal'] = pd.Series([i[5] for i in timeFrame])
    df['sellDecimal'] = pd.Series([i[6] for i in timeFrame])
    
    df['vwapAvg'] = df['vwap'].cumsum() / (df.index + 1)
    df['uppervwapAvg'] = df['STDEV_25'].cumsum() / (df.index + 1)
    df['lowervwapAvg'] = df['STDEV_N25'].cumsum() / (df.index + 1)
    
    df['derivative'] = df_dx
    
    df['buySellDif'] = pd.Series([i[2]-i[3] for i in timeFrame])
    
    #tradeTimes = [i[6] for i in AllTrades][::-1]
    epochTimes = [i[2] for i in AllTrades]
    vplist = []
    valist = []
    tpo = []
    clusterList = []
    
    for i in range(len(df['time'])-1):
        #hs = historV2(df[:i+1],50,{},AllTrades[:(len(tradeTimes) - 1 - tradeTimes.index(df['time'][i+1])) +1],[])
        hs = historV2(df[:i+1],50,{},AllTrades[:bisect.bisect_left(epochTimes, int(df['timestamp'][i+1]))],[])
        
        #time_diffs = [abs((datetime.strptime(time, '%H:%M:%S') - datetime.strptime(df['time'][i+1], '%H:%M:%S')).total_seconds()) for time in tradeTimes]
    
        # Find the index of the minimum difference
        #closest_index = time_diffs.index(min(time_diffs))
        #hs = historV2(df[:i+1],50,{},AllTrades[:(len(tradeTimes) - 1 - closest_index) +1],[])
    
        vplist.append([[xx[0], xx[3], xx[1], xx[7], xx[8]] for xx in hs[0]])
        valist.append(valueAreaV1(hs[0]))
        
        mTrade = [i for i in AllTrades[:bisect.bisect_left(epochTimes, int(df['timestamp'][i+1]))]]  #AllTrades[:(len(tradeTimes) - 1 - tradeTimes.index(df['time'][i+1])) +1]
        
         
        mTrade = sorted(mTrade, key=lambda d: d[1], reverse=True)
        
        for tdd in range(len(mTrade)):
            mTrade[tdd][4] = tdd
            
        #[mTrade[i].insert(4,i) for i in range(len(mTrade))] 
        
        newwT = []
        for x in mTrade[:100]:
            newwT.append([x[0],x[1],x[2],x[5], x[4],x[3],x[6], df['timestamp'].searchsorted(x[2])-1])
            
        
            
        tpo.append([[i[0], i[1], i[2], i[3], i[4], i[6], i[7]] for i in newwT[:100]])
        
        data = [i[0] for i in newwT[:100]]
        data.sort(reverse=True)
        differences = [abs(data[i+1] - data[i]) for i in range(len(data) - 1)]
        try:
            average_difference = sum(differences) / len(differences)
            cdata = find_clusters(data, average_difference)
            clust = [i for i in cdata if len(i) >= 5]
            clusterList.append(clust)
        except(ZeroDivisionError):
            clusterList.append([])
        #break
    else:
        hs = historV2(df,50,{},AllTrades,[])
        vplist.append([[xx[0], xx[3], xx[1], xx[7], xx[8]] for xx in hs[0]])
        valist.append(valueAreaV1(hs[0]))
        
        mTrade = [i for i in AllTrades]
        
         
        mTrade = sorted(mTrade, key=lambda d: d[1], reverse=True)
        
        for tdd in range(len(mTrade)):
            mTrade[tdd][4] = tdd
            
        #[mTrade[i].insert(4,i) for i in range(len(mTrade))] 
        
        newwT = []
        for x in mTrade[:100]:
            newwT.append([x[0],x[1],x[2],x[5], x[4],x[3],x[6], df['timestamp'].searchsorted(x[2])-1])
            
        
            
        tpo.append([[i[0], i[1], i[2], i[3], i[4], i[6], i[7]] for i in newwT[:100]])
        
        data = [i[0] for i in newwT[:100]] #150
        data.sort(reverse=True)
        differences = [abs(data[i+1] - data[i]) for i in range(len(data) - 1)]
        average_difference = sum(differences) / len(differences)
        cdata = find_clusters(data, average_difference)
        clust = [i for i in cdata if len(i) >= 5] #6
        clusterList.append(clust)
            
    '''
    imbalance = []
    for i in range(len(df['buyDecimal'])):
        if df['buyDecimal'][i] >= 0.58:
            mn = df['buyCount'].loc[:i].mean()
            if df['buyCount'][i] >= mn:
                imbalance.append(1)
            else:
                imbalance.append(0)
        elif df['sellDecimal'][i] >= 0.58:
            mn = df['sellCount'].loc[:i].mean()
            if df['sellCount'][i] >= mn:
                imbalance.append(2)
            else:
                imbalance.append(0)
        else:
            imbalance.append(0)
            
    df['imbalance'] = pd.Series([i for i in imbalance])
    '''
    
    df['LowVA'] = pd.Series([i[0] for i in valist])
    df['HighVA'] = pd.Series([i[1] for i in valist])
    df['POC']  = pd.Series([i[2] for i in valist])
    df['indes'] = pd.Series([i for i in range(0,len(df))])
    #---------------------------------------------------------------    
    finalTpo = []
    newTPO = []
    for i in tpo:
        for x in i:
            newTPO.append(x[0])
            newTPO.append(x[1])
            newTPO.append(x[6])
            if x[3] == 'A':
               newTPO.append(0)
            elif x[3] == 'B':
               newTPO.append(1)
            else:
               newTPO.append(2)
            newTPO.append(x[4])
        finalTpo.append(newTPO)
        newTPO = []
        #print(i)
        #break
        
    max_length = max(len(inner_list) for inner_list in finalTpo)
    for inner_list in finalTpo:
        while len(inner_list) < max_length:
            inner_list.append(0)
    
    # Determine the number of columns
    max_columns = max(len(row) for row in finalTpo)
    # Generate column names
    column_names = [f"TopOrders_{i}" for i in range(max_columns)]
    # Create a DataFrame
    df1 = pd.DataFrame(finalTpo, columns=column_names)
    df= pd.concat([df, df1],  axis = 1)
    #---------------------------------------------------------------
    finalClust = []
    newClust = []
    for i in range(len(clusterList)):
        for c in sorted(clusterList[i], key=len, reverse=True):
            newClust.append(c[0])
            newClust.append(c[len(c)-1])
            newClust.append(len(c))
            
            bidCount = 0
            askCount = 0
            midCount = 0
            for tp in tpo[i]:
                if c[len(c)-1] <= tp[0] <= c[0] :
                    if tp[3] == 'B':
                        bidCount+= tp[1]
                    elif tp[3] == 'A':
                        askCount+= tp[1]
                    elif tp[3] == 'N':
                        midCount+= tp[1]
                        
            newClust.append(bidCount)
            newClust.append(askCount)
            #newClust.append(midCount)
            newClust.append(askCount+bidCount+midCount)
            newClust.append(bidCount/ (bidCount+askCount+midCount+1))
            newClust.append(askCount/ (bidCount+askCount+midCount+1))
            
        finalClust.append(newClust)
        newClust = []
        
    max_length = max(len(inner_list) for inner_list in finalClust)
    
    # Pad each inner list with zeros until it reaches the maximum length
    for inner_list in finalClust:
        while len(inner_list) < max_length:
            inner_list.append(0)
            
    
    column_names = [f"Cluster{i}" for i in range(max_length)]
    # Create a DataFrame
    df1 = pd.DataFrame(finalClust, columns=column_names)
    df= pd.concat([df, df1],  axis = 1)
    #---------------------------------------------------------------
    finalvp = []
    newvp = []
    for i in vplist:
        for v in i:
            newvp.append(v[0])
            newvp.append(v[1])
            newvp.append(v[2])
            newvp.append(v[3])
            newvp.append(v[4])
        finalvp.append(newvp)
        newvp = []
        
        
    max_length = max(len(inner_list) for inner_list in finalvp)
    
    # Pad each inner list with zeros until it reaches the maximum length
    for inner_list in finalvp:
        while len(inner_list) < max_length:
            inner_list.append(0)
    
    column_names = [f"VolPro{i}" for i in range(max_length)]
    # Create a DataFrame
    df1 = pd.DataFrame(finalvp, columns=column_names)
    df= pd.concat([df, df1],  axis = 1)
    
    
    #---------------------------------------------------------------
    fbuyss = []
    fsellss = []
    
    for indx in range(len(df['indes'])):
            
        buys = [i for i in df['buyCount'].iloc[:indx+1]]
        sells = [i for i in df['sellCount'].iloc[:indx+1]]
        
        
        fbuyss.append(sum(buys))
        fsellss.append(sum(sells))
    
    df1 = pd.DataFrame([[fbuyss[i],fsellss[i]] for i in range(len(fbuyss))], columns=['buyCountCum', 'sellCountCum'])
    df= pd.concat([df, df1],  axis = 1)
    #-----------------------------------------------------------------------------------------------------------
    fbuyss = []
    fsellss = []
    
    for indx in range(len(df['indes'])):
            
        buys = [i for i in df['buyCount'].iloc[:indx+1]]
        sells = [i for i in df['sellCount'].iloc[:indx+1]]
        
        
            
        newBuys = [abs(buys[i]-sells[i]) for i in range(len(buys)) if buys[i]-sells[i] > 0]
        newSells = [abs(buys[i]-sells[i]) for i in range(len(buys)) if buys[i]-sells[i] < 0]
        
        buySum = sum(newBuys)
        sellSum = sum(newSells)
        
        fbuyss.append(buySum)
        fsellss.append(sellSum)
    
    df1 = pd.DataFrame([[fbuyss[i],fsellss[i]] for i in range(len(fbuyss))], columns=['buyDiffSum', 'sellDiffSum'])
    df= pd.concat([df, df1],  axis = 1)
    
    
    
    #--------------------------------------------------------------------------------------------------------------
    localMin = []
    localMax = []
    for sf in range(len(df)):
        localMin.append([df['low'][i] for i in argrelextrema(df[:sf+1].low.values, np.less_equal, order=18)[0]])
        localMax.append([df['high'][i] for i in argrelextrema(df[:sf+1].high.values, np.greater_equal, order=18)[0]])


    max_length = max(len(inner_list) for inner_list in localMin)
    
    for inner_list in localMin:
        while len(inner_list) < max_length:
            inner_list.append(0)
            
    column_names = [f"localMin{i}" for i in range(max_length)]
    # Create a DataFrame
    df1 = pd.DataFrame(localMin, columns=column_names)
    df= pd.concat([df, df1],  axis = 1)
    
    #prevDf = pd.read_csv(stkName+'Data-4.csv')
    blob = Blob('Daily'+stkName, bucket)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)

    prevDf = pd.read_csv(buffer)
    
    prevDf_min_cols = [col for col in prevDf.columns if col.startswith('localMin')]
    cuurtmincols = [col for col in df.columns if col.startswith('localMin')]
    
    if len(prevDf_min_cols) > len(cuurtmincols):
        for cl in prevDf_min_cols:
            if cl not in cuurtmincols:
                df[cl] = 0
                cuurtmincols.append(cl)
                
    elif len(cuurtmincols) > len(prevDf_min_cols):
        for cl in cuurtmincols:
            if cl not in prevDf_min_cols:
                prevDf[cl] = 0
                prevDf_min_cols.append(cl)

    
    #---------------------------------------------------------------
    max_length = max(len(inner_list) for inner_list in localMax)
    
    for inner_list in localMax:
        while len(inner_list) < max_length:
            inner_list.append(0)
            
    column_names = [f"localMax{i}" for i in range(max_length)]
    # Create a DataFrame
    df1 = pd.DataFrame(localMax, columns=column_names)
    df= pd.concat([df, df1],  axis = 1)
    
    
    prevDf_min_cols = [col for col in prevDf.columns if col.startswith('localMax')]
    cuurtmincols = [col for col in df.columns if col.startswith('localMax')]
    
    if len(prevDf_min_cols) > len(cuurtmincols):
        for cl in prevDf_min_cols:
            if cl not in cuurtmincols:
                df[cl] = 0
                cuurtmincols.append(cl)
                
    elif len(cuurtmincols) > len(prevDf_min_cols):
        for cl in cuurtmincols:
            if cl not in prevDf_min_cols:
                prevDf[cl] = 0
                prevDf_min_cols.append(cl)
               
    #---------------------------------------------------------------
    
    lfbuySum = []
    lfsellSum = []
    
    for indx in range(len(df['indes'])):
        if indx - 4 < 0:
            lfbuySum.append(sum(df['buyCount'][:indx+1]))
            lfsellSum.append(sum(df['sellCount'][:indx+1]))
        elif indx - 4 >= 0:
            lfbuySum.append(sum(df['buyCount'][indx-4:indx+1]))
            lfsellSum.append(sum(df['sellCount'][indx-4:indx+1]))
            
    df1 = pd.DataFrame(lfbuySum, columns=['buyCount5C'])
    df = pd.concat([df, df1],  axis = 1)
    
    df1 = pd.DataFrame(lfsellSum, columns=['sellCount5C'])
    df = pd.concat([df, df1],  axis = 1)
            
    
    #---------------------------------------------------------------
    
    finall = []
    for i in tpo:
        tobuyss =  sum([t[1] for t in i if t[3] == 'B'])
        tosellss = sum([t[1] for t in i if t[3] == 'A'])
        
        try:
            finall.append([tobuyss,tosellss, tobuyss/(tobuyss+tosellss), tosellss/(tobuyss+tosellss)])
        except(ZeroDivisionError):
            finall.append([0,0, 0, 0])
        
    df1 = pd.DataFrame(finall, columns=['topOrderBuy', 'topOrderSell', 'topOrderBuyPercent', 'topOrderSellPercent'])
    df = pd.concat([df, df1],  axis = 1)
    
    print(stkName)
    #df.to_csv(stkName+'Data-3.csv', index=False)
    #prevDf = pd.read_csv(stkName+'Data-2.csv')
    
    if prevDf.shape[1] > df.shape[1]:
        pattern = 'Cluster'
        matching_columns = [col for col in df.columns if col.startswith(pattern)]
        last_matching_column = matching_columns[-1]
        colCount = int(last_matching_column.replace('Cluster',''))+1
        while prevDf.shape[1] > df.shape[1]:
            df['Cluster'+str(colCount)] = 0
            colCount+=1
            
        col_cluster_cols = [col for col in df.columns if col.startswith('Cluster')]
        col_vp_cols = [col for col in df.columns if col.startswith('VolPro')]
        
        # Identify other columns
        other_cols = [col for col in df.columns if col not in col_cluster_cols + col_vp_cols] 
        
        # Rearrange columns
        new_column_order = other_cols+col_cluster_cols+col_vp_cols
        
        # Reorder the DataFrame
        df = df[new_column_order]
        combined_df = pd.concat([prevDf, df], ignore_index=True)
        #combined_df.to_csv(stkName+'Data-4.csv', index=False)  #mode='a',
        #df.to_csv(stkName+'Data-1.csv',mode='a', index=False) 
        
    elif df.shape[1] > prevDf.shape[1]:
        pattern = 'Cluster'
        matching_columns = [col for col in prevDf.columns if col.startswith(pattern)]
        last_matching_column = matching_columns[-1]
        colCount = int(last_matching_column.replace('Cluster',''))+1
        while df.shape[1] > prevDf.shape[1]:
            prevDf['Cluster'+str(colCount)] = 0
            colCount+=1
            
        col_cluster_cols = [col for col in prevDf.columns if col.startswith('Cluster')]
        col_vp_cols = [col for col in prevDf.columns if col.startswith('VolPro')]
        
        other_cols = [col for col in prevDf.columns if col not in col_cluster_cols + col_vp_cols]
        
        # Rearrange columns
        new_column_order = other_cols+col_cluster_cols+col_vp_cols
        
        # Reorder the DataFrame
        prevDf = prevDf[new_column_order]
        
        combined_df = pd.concat([prevDf, df], ignore_index=True)
        #combined_df.to_csv(stkName+'Data-4.csv', index=False) 
        
        # Display the DataFrame with the new column order
        #print(df.head())
    elif df.shape[1] == prevDf.shape[1]:
        
        combined_df = pd.concat([prevDf, df], ignore_index=True)
        #combined_df.to_csv(stkName+'Data-4.csv', index=False) 
        
        
    df = combined_df

    fig = go.Figure(data=[go.Candlestick(x=pd.Series([i for i in range(len(df))]),
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])


    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['POC'], mode='lines',name='POC',marker_color='#0000FF'))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['100ema'], mode='lines', opacity=0.50, name='100ema',marker_color='rgba(0,0,0)'))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['200ema'], mode='lines', opacity=0.50,name='200ema',marker_color='rgba(0,0,0)'))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['50ema'], mode='lines', opacity=0.50,name='50ema',marker_color='rgba(0,0,0)'))
    
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_2'], mode='lines', opacity=0.1, name='UPPERVWAP2', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_N2'], mode='lines', opacity=0.1, name='LOWERVWAP2', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_25'], mode='lines', opacity=0.15, name='UPPERVWAP2.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_N25'], mode='lines', opacity=0.15, name='LOWERVWAP2.5', line=dict(color='black')))
   
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_1'], mode='lines', opacity=0.1, name='UPPERVWAP1', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_N1'], mode='lines', opacity=0.1, name='LOWERVWAP1', line=dict(color='black')))
            
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_15'], mode='lines', opacity=0.1, name='UPPERVWAP1.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_N15'], mode='lines', opacity=0.1, name='LOWERVWAP1.5', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_0'], mode='lines', opacity=0.1, name='UPPERVWAP0.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=pd.Series([i for i in range(len(df))]), y=df['STDEV_N0'], mode='lines', opacity=0.1, name='LOWERVWAP0.5', line=dict(color='black')))

    # Update layout
    fig.update_layout(title=stkName+' Chart '+ str(datetime.now().time()),
                      xaxis_title='Time',
                      yaxis_title='Price',
                      height=700,
                      xaxis_rangeslider_visible=False,
                      xaxis=dict(range=[int(len(df)*0.83), len(df)]),
                      yaxis=dict(range=[min([i for i in df['STDEV_ N25'][int(len(df)*0.83):len(df)]]), max([i for i in df['STDEV_25'][int(len(df)*0.83):len(df)]])])) #showlegend=False
    
    #fig.update_layout(
    ##xaxis=dict(range=[2, 4]),  # Zoom in on x-axis between 2 and 4
      # Zoom in on y-axis between 11 and 13


    # Show the chart
    #fig.show() 
    
    return fig

if __name__ == '__main__': 
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
        
      
        
        
     
    
    
      


