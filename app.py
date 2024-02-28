# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:41:52 2024

@author: UOBASUB
"""
from google.cloud.storage import Blob
from google.cloud import storage
from datetime import datetime, timedelta, date, time
import csv
import io
import pandas as pd
import numpy as np
import math
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
from scipy.signal import argrelextrema



def ema(df):
    df['30ema'] = df['close'].ewm(span=30, adjust=False).mean()
    df['40ema'] = df['close'].ewm(span=40, adjust=False).mean()
    df['28ema'] = df['close'].ewm(span=28, adjust=False).mean()
    df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
    df['100ema'] = df['close'].ewm(span=50, adjust=False).mean()
    df['1ema'] = df['close'].ewm(span=1, adjust=False).mean()

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
        val = df.volume2Sum / (df.volumeSum+0.0000000000001) - df.vwap * df.vwap
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


def historV1(name, num, quodict, trad:list=[], quot:list=[], rangt:int=1):
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
        i+=countCandle(trad,[],i[0],i[3],name,{})

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

symbolNumList = ['17077', '750', '44740', '31863', '204839', '42007360', '7062', '2259', '156627', '156755', '1545', '449', '270851', '4130291', '950', '187593', '363390']
symbolNameList = ['ESH4','NQH4', 'GCJ4', 'HGH4', 'YMH4', 'BTCG4', 'RTYH4', '6NH4', '6EH4', '6AH4', '6CH4', 'SIH4', 'CLJ4', 'ZFH4', 'NGH4', 'TNH4', 'UBH4']

gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage")

from dash import Dash, dcc, html, Input, Output, callback, State
inter = 60000#60000
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
    html.Div(id='container-button-basic',children="Enter a symbol from |'ESH4' 'NQH4' 'CLH4' 'GCJ4' 'HGH4' 'YMH4' 'BTCG4' 'RTYH4'| and submit"),
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
        return 'The input symbol was '+str(value)+" is not accepted please try different symbol from  |'ESH4' 'NQH4' 'CLH4' 'GCJ4' 'HGH4' 'YMH4' 'BTCG4' 'RTYH4'|  ", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(Output('graph', 'figure'),
          Input('interval', 'n_intervals'),
          State('stkName-value', 'data'))


    
def update_graph_live(n_intervals, data):
    print('inFunction')	

    if data in symbolNameList:
        stkName = data
        symbolNum = symbolNumList[symbolNameList.index(stkName)] 
    else:
        stkName = 'NQH4'  
        symbolNum = symbolNumList[symbolNameList.index(stkName)]


    
    
    blob = Blob('5dTrades'+symbolNum, bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    fdTrades = [i for i in csv_rows]
    
    for sublist in fdTrades:
        if len(sublist[0]) != 19:
            sublist[0] = sublist[0] + '0' * (19 - len(sublist[0]))
    
    blob = Blob('FuturesTrades'+str(symbolNum), bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
    
    STrades = [i for i in csv_rows]
    
    for i in STrades:
        i[1] = str(int(i[1])/1e9)
    
    
    allTrades = fdTrades + STrades
    
    allTrades = sorted(allTrades, key=lambda d: int(d[2]), reverse=True)
    
    allTrades = allTrades[0:200]
    
    newAllTrades = []
    for i in allTrades:
        opttimeStamp = datetime.utcfromtimestamp(int(i[0])/ 1e9).strftime("%Y-%m-%d %H:%M:%S")
        newAllTrades.append([float(i[1]), int(i[2]), int(i[0]), 0, i[3], opttimeStamp])
     
        
    [newAllTrades[i].insert(4,i) for i in range(len(newAllTrades))] 
    
    
    newwT = []
    for i in newAllTrades:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
        
    #------------------------------------------------------------------------------
    
    blob = Blob('5dOHLC'+symbolNum, bucket) 
    FuturesFDOHLC = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesFDOHLC))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    fdOHLC = [i for i in csv_rows]
    
    blob = Blob('FuturesOHLC'+str(symbolNum), bucket) 
    FuturesOHLC = blob.download_as_text()
        
    
    csv_reader  = csv.reader(io.StringIO(FuturesOHLC))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    cOHLC = [i for i in csv_rows]
    
    for i in cOHLC:
        i[2] = str(int(i[2])/1e9)
        i[3] = str(int(i[3])/1e9)
        i[4] = str(int(i[4])/1e9)
        i[5] = str(int(i[5])/1e9)
    
    
    df = pd.DataFrame(cOHLC, columns = ['Timestamp', 'symbolNum', 'open', 'high', 'low', 'close', 'volume'])
    
    df['strTime'] = df['Timestamp'].apply(lambda x: pd.Timestamp(int(x) // 10**9, unit='s') )
    df['strTime'] = pd.to_datetime(df['strTime'])
    df.set_index('strTime', inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    df_resampled = df.resample('5T').agg({
        'Timestamp': 'first',
        'symbolNum': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
        
    allOHLC = fdOHLC + df_resampled.values.tolist()
    
    aggs = []
    for i in allOHLC:
        try:
            '''
            hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
            if hourss < 10:
                hourss = '0'+str(hourss)
            minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
            if minss < 10:
                minss = '0'+str(minss)
            opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
            '''
            opttimeStamp = pd.to_datetime(int(i[0]), unit='ns').strftime("%Y-%m-%d %H:%M:%S")
            aggs.append([float(i[2]), float(i[3]), float(i[4]), float(i[5]), int(i[6]), opttimeStamp, int(i[0]), int(i[1])])
        except(TypeError, ValueError):
            pass
            
        
    newAggs = []
    for i in aggs:
        if i not in newAggs:
            newAggs.append(i)
            
          
    df = pd.DataFrame(newAggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',])   
    
    
    vwap(df)
    ema(df)
    PPP(df)
    
    
    
    
    fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], mode='lines', name='VWAP'))
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', opacity=0.15, name='UPPERVWAP2', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N2'], mode='lines', opacity=0.15, name='LOWERVWAP2', line=dict(color='black')))
    #if 0 in lstVwap:
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_25'], mode='lines', opacity=0.15, name='UPPERVWAP2.5', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N25'], mode='lines', opacity=0.15, name='LOWERVWAP2.5', line=dict(color='black')))
    #if 1 in lstVwap:    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_1'], mode='lines', opacity=0.15, name='UPPERVWAP1', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N1'], mode='lines', opacity=0.15, name='LOWERVWAP1', line=dict(color='black')))
        
    #if 1.5 in lstVwap:     
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_15'], mode='lines', opacity=0.15, name='UPPERVWAP1.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N15'], mode='lines', opacity=0.15, name='LOWERVWAP1.5', line=dict(color='black')))
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_0'], mode='lines', opacity=0.15, name='UPPERVWAP0.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N0'], mode='lines', opacity=0.15, name='LOWERVWAP0.5', line=dict(color='black')))
    
    
    

    blob = Blob('PrevDay', bucket) 
    PrevDay = blob.download_as_text()
        

    csv_reader  = csv.reader(io.StringIO(PrevDay))

    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
        
    previousDay = [csv_rows[[i[4] for i in csv_rows].index(symbolNum)][0] ,csv_rows[[i[4] for i in csv_rows].index(symbolNum)][1] ,csv_rows[[i[4] for i in csv_rows].index(symbolNum)][2]]
    
    

    if len(previousDay) > 0:
        fig.add_trace(go.Scatter(x=df['time'],
                                 y= [float(previousDay[2])]*len(df['time']) ,
                                 line_color='cyan',
                                 text = str(previousDay[2]),
                                 textposition="bottom left",
                                 name='PreviousDay POC '+ str(previousDay[2]),
                                 showlegend=False,
                                 visible=False,
                                 mode= 'lines',
                                ),
                     )

        fig.add_trace(go.Scatter(x=df['time'],
                                 y= [float(previousDay[0])]*len(df['time']) ,
                                 line_color='green',
                                 text = str(previousDay[0]),
                                 textposition="bottom left",
                                 name='Previous LVA '+ str(previousDay[0]),
                                 showlegend=False,
                                 visible=False,
                                 mode= 'lines',
                                ),
                     )

        fig.add_trace(go.Scatter(x=df['time'],
                                 y= [float(previousDay[1])]*len(df['time']) ,
                                 line_color='purple',
                                 text = str(previousDay[1]),
                                 textposition="bottom left",
                                 name='Previous HVA '+ str(previousDay[1]),
                                 showlegend=False,
                                 visible=False,
                                 mode= 'lines',
                                ),
                     )

    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', opacity=0.15, name='1ema', line=dict(color='black')))
    

    sortadlist = newwT[:40]
    for v in range(len(sortadlist)):
        fig.add_trace(go.Scatter(x=df['time'],
                                 y= [sortadlist[v][0]]*len(df['time']) ,
                                 line_color = 'rgb(0,104,139)' if (str(sortadlist[v][3]) == 'B') else 'brown' if (str(sortadlist[v][3]) == 'A') else 'rgb(0,0,0)',
                                 text = str(sortadlist[v][4]) + ' ' + str(sortadlist[v][1]) + ' ' + str(sortadlist[v][3])  + ' ' + str(sortadlist[v][6]),
                                 #text='('+str(priceDict[sortadlist[v][0]]['ASKAVG'])+'/'+str(priceDict[sortadlist[v][0]]['BIDAVG']) +')'+ '('+str(priceDict[sortadlist[v][0]]['ASK'])+'/'+str(priceDict[sortadlist[v][0]]['BID']) +')'+  '('+ sortadlist[v][3] +') '+str(sortadlist[v][4]),
                                 textposition="bottom left",
                                 name=str(sortadlist[v][0]),
                                 showlegend=False,
                                 visible=False,
                                 mode= 'lines',
                                
                                ),
                     )
        
    for trd in newwT[:50]:
        trd.append(df['timestamp'].searchsorted(trd[2])-1)
        
    
    for trds in newwT[:50]:
        try:
            if str(trds[3]) == 'A':
                vallue = 'sell'
                sidev = trds[0]
            elif str(trds[3]) == 'B':
                vallue = 'buy'
                sidev = trds[0]
            else:
                vallue = 'Mid'
                sidev = df['open'][trds[7]]
            fig.add_annotation(x=df['time'][trds[7]], y=sidev,
                               text= str(trds[4]) + ' ' + str(trds[1]) + ' ' + vallue ,
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
        except(KeyError):
            continue 

    '''
    localMin = argrelextrema(df.close.values, np.less_equal, order=120)[0] 
    localMax = argrelextrema(df.close.values, np.greater_equal, order=120)[0]
    
    if len(localMin) > 0:
        mcount = 0 
        for p in localMin:
            fig.add_annotation(x=df['time'][p], y=df['close'][p],
                            text= str(mcount) +'Min' ,
                            showarrow=True,
                            arrowhead=4,
                            font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
            mcount+=1
    if len(localMax) > 0:
        mcount = 0 
        for b in localMax:
            fig.add_annotation(x=df['time'][b], y=df['close'][b],
                            text=str(mcount) + 'Max',
                            showarrow=True,
                            arrowhead=4,
                            font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
            mcount+=1
    '''   
        
    for tmr in range(0,len(fig.data)): 
        fig.data[tmr].visible = True
        
    steps = []
    for i in np.arange(0,len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            #label=str(pricelist[i-1])
        )
        for u in range(0,i):
            step["args"][0]["visible"][u] = True
            
        
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    #print(steps)
    #if previousDay:
        #nummber = 6
    #else:
        #nummber = 0
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Price: "},
        pad={"t": 10},
        steps=steps[12:]#[8::3]
    )]
    
    fig.update_layout(
        sliders=sliders
    )
        
    
    
    
    # Update layout
    fig.update_layout(title=stkName + ' '+  str(datetime.now()),
                      yaxis_title='Price',
                      xaxis=dict(type = "category"),
                      xaxis_rangeslider_visible=False, showlegend=False, height=800,)
    
    fig.update_xaxes(showticklabels=False)
    
    return fig


if __name__ == '__main__': 
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
            