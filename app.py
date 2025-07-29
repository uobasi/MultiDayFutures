# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:30:08 2025

@author: uobas
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:58:00 2025

@author: uobas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:24:13 2025

@author: UOBASUB
"""

#import csv
import io
from datetime import datetime, timedelta, date, time
import pandas as pd 
import numpy as np
#import math
from google.cloud.storage import Blob
from google.cloud import storage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None
#from scipy.signal import argrelextrema
#from scipy import signal
#from scipy.misc import derivative
#from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import plotly.io as pio
pio.renderers.default='browser'
import bisect
#from collections import defaultdict
#import gcsfs
import re
from concurrent.futures import ThreadPoolExecutor  
from io import StringIO  
from scipy.stats import percentileofscore
#import gc
from collections import Counter
from collections import defaultdict


def download_data(bucket_name, blob_name):
    blob = Blob(blob_name, bucket_name)
    return blob.download_as_text()   

# Function to extract the first date occurrence
def extract_first_date(filename):
    # Regex to extract the first date occurrence (_YYYY-MM-DD_)
    date_pattern = re.compile(r"_(\d{4}-\d{2}-\d{2})_")
    match = date_pattern.search(filename)  # Find the first match
    return match.group(1) if match else "0000-00-00"  # Default for sorting safety



def historV2(df, num, quodict, trad:list=[], quot:list=[]):
    trad_array = np.asarray(trad, dtype=object)  # Convert once, avoid reallocation

    # **Sort trades by price (Much faster way)**
    sorted_indices = np.argsort(trad_array[:, 0], kind='stable')  # Use stable sorting
    sorted_trades = trad_array[sorted_indices]

    # **Efficiently Extract Unique Prices and Sum Their Volumes**
    unique_prices, index = np.unique(sorted_trades[:, 0], return_index=True)
    summed_volumes = np.add.reduceat(sorted_trades[:, 1].astype(int), index)

    # **Find Point of Control (POC) (Price with max volume)**
    pocT = unique_prices[np.argmax(summed_volumes)]

    # **Create histogram bins from unique prices instead of full data**
    hist, bin_edges = np.histogram(unique_prices, bins=num)

    # **Preallocate output arrays (Avoid slow appends)**
    cptemp = np.zeros((len(hist), 4), dtype=object)

    # **Vectorized Bin Searching**
    start_indices = np.searchsorted(unique_prices, bin_edges[:-1], side='left')
    #end_indices = np.searchsorted(unique_prices, bin_edges[1:], side='right')

    # **Process each bin using NumPy vectorized operations**
    valid_indices = start_indices[start_indices < len(summed_volumes)]

    # **Compute sum of volumes in each bin**
    bin_sums = np.zeros(len(hist), dtype=int)
    bin_sums[:len(valid_indices)] = np.add.reduceat(summed_volumes, valid_indices)

    # **Assign bin data without looping**
    cptemp[:, 0] = bin_edges[:-1]  # Start of bin
    cptemp[:, 1] = bin_sums        # Summed volume in bin
    cptemp[:, 2] = np.arange(len(hist))  # Index
    cptemp[:, 3] = bin_edges[1:]   # End of bin

    return [cptemp.tolist(), pocT]


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

def historV1(df, num, quodict, trad:list=[], quot:list=[]):
    trad_array = np.array(trad, dtype=object)  # Ensure consistency
    
    sorted_indices = np.argsort(trad_array[:, 0])
    sorted_trades = trad_array[sorted_indices]
    
    unique_prices, index = np.unique(sorted_trades[:, 0], return_index=True)

    # Compute summed volume for each unique price
    summed_volumes = np.add.reduceat(sorted_trades[:, 1].astype(int), index)
    # Extract unique prices and counts efficiently using NumPy
    #unique_prices, price_counts = np.unique(trad_array[:, 0], return_counts=True)
    
    # Point of Control (POC) - Price with the highest traded volume
    pocT = unique_prices[np.argmax(summed_volumes)]

    # Sort trades by price **only once**
    #sorted_indices = trad_array[:, 0].argsort()
    #mTradek = trad_array[sorted_indices]  # Sorted by price
    priceList = sorted_trades[:, 0]  # Extract sorted price column

    # Use NumPy to create bins
    hist, bin_edges = np.histogram(unique_prices, bins=num)

    # Preallocate arrays (Much faster than appending lists)
    cptemp = np.zeros((len(hist), 4), dtype=object)  # Store bin data
    zipList = np.zeros((len(hist), 3), dtype=int)    # Store buy/sell/neutral counts

    # Vectorized bin searching
    start_indices = np.searchsorted(priceList, bin_edges[:-1], side='left')
    end_indices = np.searchsorted(priceList, bin_edges[1:], side='right')

    # Process each bin efficiently
    for i in range(len(hist)):
        bin_trades = sorted_trades[start_indices[i]:end_indices[i]]

        if len(bin_trades) > 0:
            # Efficient volume sum calculations
            pziCount = np.sum(bin_trades[:, 1].astype(int))

            # Efficient buy/sell/neutral counts using np.where
            #acount = np.sum(bin_trades[:, 1][np.where(bin_trades[:, 5] == 'A')].astype(int))
            #bcount = np.sum(bin_trades[:, 1][np.where(bin_trades[:, 5] == 'B')].astype(int))
            #ncount = np.sum(bin_trades[:, 1][np.where(bin_trades[:, 5] == 'N')].astype(int))

            cptemp[i] = [bin_edges[i], pziCount, i, bin_edges[i+1]]
            #zipList[i] = [acount, bcount, ncount]

    # Convert list for further processing (avoiding slow append operations)
    cptemp = cptemp.tolist()

    # Append countCandle() results (Still needs to loop, but optimized)
    for i in range(len(cptemp)):
        cptemp[i] += countCandle(trad, [], cptemp[i][0], cptemp[i][3], df['name'].iloc[0], {})

    # Merge zipList into cptemp
    cptemp = [cptemp[i] + zipList[i].tolist() for i in range(len(cptemp))]

    # Sort results by total volume (Final sorting step)
    sortadlist = sorted(cptemp, key=lambda stock: float(stock[1]), reverse=True)

    return [cptemp, sortadlist, pocT]


def valueAreaV3(lst):
    # Ensure list is not empty
    if not lst:
        return [None, None, None]

    # Filter out entries with zero volume
    mkk = [i for i in lst if i[1] > 0]
    if not mkk:
        mkk = lst

    # Assign indices for tracking
    for idx, item in enumerate(mkk):
        item[2] = idx

    # Total volume in mkk
    total_volume = sum([i[1] for i in mkk])
    if total_volume == 0:
        return [None, None, None]

    # Identify POC (Point of Control) by maximum volume
    poc_item = max(mkk, key=lambda x: x[1])
    pocIndex = poc_item[2]
    sPercent = total_volume * 0.70  # 70% of total volume
    accumulated_volume = poc_item[1]  # Start with POC volume

    # Initialize Value Area boundaries
    topIndex, dwnIndex = pocIndex, pocIndex

    # Expand the value area until 70% of volume is captured
    while accumulated_volume < sPercent:
        topVol = mkk[topIndex - 1][1] if topIndex > 0 else 0
        dwnVol = mkk[dwnIndex + 1][1] if dwnIndex < len(mkk) - 1 else 0

        # Add the larger volume to the total and adjust indices
        if topVol >= dwnVol:
            if topIndex > 0:
                topIndex -= 1
                accumulated_volume += topVol
        else:
            if dwnIndex < len(mkk) - 1:
                dwnIndex += 1
                accumulated_volume += dwnVol

        # Break if boundaries are fully expanded
        if topIndex == 0 and dwnIndex == len(mkk) - 1:
            break

    # Return Value Area Low, Value Area High, and POC
    return [mkk[topIndex][0], mkk[dwnIndex][0], poc_item[0]]


def combine_histogram_data(hist1, hist2):
    # Extract the first list and POC from each set
    hist1_data, poc1 = hist1
    hist2_data, poc2 = hist2
    
    # Extract the price range from both histograms
    min_price = min(hist1_data[0][0], hist2_data[0][0])
    max_price = max(hist1_data[-1][3], hist2_data[-1][3])
    
    # Create 100 evenly spaced bins over the combined range
    bin_width = (max_price - min_price) / 100
    new_bins = []
    
    for i in range(100):
        start_price = min_price + i * bin_width
        end_price = min_price + (i + 1) * bin_width
        
        # Initialize the new bin with zero volume
        new_bins.append([start_price, 0, i, end_price])
    
    # Function to distribute volume from one histogram to new bins
    def distribute_volumes(hist_data):
        for bin_data in hist_data:
            bin_start = bin_data[0]
            bin_end = bin_data[3]
            bin_volume = bin_data[1]
            
            # Find the overlapping new bins and distribute volume proportionally
            for i, new_bin in enumerate(new_bins):
                new_start = new_bin[0]
                new_end = new_bin[3]
                
                # Check for overlap
                overlap_start = max(bin_start, new_start)
                overlap_end = min(bin_end, new_end)
                
                if overlap_start < overlap_end:
                    # Calculate the proportion of the original bin that overlaps with the new bin
                    overlap_width = overlap_end - overlap_start
                    original_width = bin_end - bin_start
                    proportion = overlap_width / original_width
                    
                    # Add proportional volume to the new bin
                    new_bins[i][1] += bin_volume * proportion
    
    # Distribute volumes from both histograms
    distribute_volumes(hist1_data)
    distribute_volumes(hist2_data)
    
    # Round volumes to integers
    for bin_data in new_bins:
        bin_data[1] = int(round(bin_data[1]))
    
    # Calculate new POC
    new_poc = new_bins[max(range(len(new_bins)), key=lambda i: new_bins[i][1])][0]
    
    return [new_bins, new_poc]



def combine_histogram_data_2(hist1, hist2, bins=100):
    """
    Combine two histograms of the form:
       hist = ([ [start, volume, idx, end], … ], poc)
    into a single `bins`-bin histogram spanning the full min→max range,
    redistributing each original bin’s volume proportionally by overlap.
    Returns (new_bins, new_poc), where
      new_bins = [ [start_i, vol_i, i, end_i], … ]
      new_poc  = midpoint of the bin with the highest combined volume
    """
    # Unpack
    data1, _ = hist1
    data2, _ = hist2

    # Find global min & max edges
    all_starts = [row[0] for row in data1] + [row[0] for row in data2]
    all_ends   = [row[3] for row in data1] + [row[3] for row in data2]
    global_min = min(all_starts)
    global_max = max(all_ends)

    # Build bin edges
    edges = np.linspace(global_min, global_max, bins+1)
    bin_starts = edges[:-1]
    bin_ends   = edges[1:]

    # Prepare accumulator
    combined_vol = np.zeros(bins, dtype=float)

    # Helper to add one histogram’s volumes
    def add_hist(data):
        for start, vol, _, end in data:
            width = end - start
            # compute overlap of [start,end] with each new bin
            overlap = np.minimum(bin_ends, end) - np.maximum(bin_starts, start)
            overlap = np.clip(overlap, 0, None)
            # fraction of this old bin that falls in each new bin
            frac = overlap / width
            combined_vol[:] += vol * frac

    # Distribute both
    add_hist(data1)
    add_hist(data2)

    # Round to int
    combined_vol = np.rint(combined_vol).astype(int)

    # Build output new_bins list
    new_bins = [
        [float(bin_starts[i]), int(combined_vol[i]), i, float(bin_ends[i])]
        for i in range(bins)
    ]

    # POC = midpoint of the bin with max volume
    poc_idx = int(combined_vol.argmax())
    poc_price = float((bin_starts[poc_idx] + bin_ends[poc_idx]) / 2)

    return [new_bins, poc_price]


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

from scipy import signal
def butter_lowpass_realtime(data, cutoff=0.05, order=2):
    """
    Apply a real-time Butterworth low-pass filter to smooth stock prices.
    Uses lfilter() with proper initial conditions.
    
    :param data: List or Pandas Series of stock prices.
    :param cutoff: Normalized cutoff frequency (0 < cutoff < 1), lower = smoother.
    :param order: Filter order (higher = sharper cutoff).
    :return: Smoothed stock price series (real-time compatible).
    """
    b, a = signal.butter(order, cutoff, btype='low', analog=False)
    
    # Set initial conditions using the first value to avoid starting at zero
    zi = signal.lfilter_zi(b, a) * data[0]
    
    # Apply the filter in a forward-only manner
    smoothed_data, _ = signal.lfilter(b, a, data, zi=zi)
    
    return smoothed_data



def find_clusters_1(data, threshold):
    if not data:
        return []

    clusters = []
    current_cluster = [data[0]]
    current_sum = data[0][1]

    for i in range(1, len(data)):
        prev_price = current_cluster[-1][0]
        curr_price = data[i][0]

        if abs(curr_price - prev_price) <= threshold:
            current_cluster.append(data[i])
            current_sum += data[i][1]
        else:
            clusters.append((current_cluster, current_sum))
            current_cluster = [data[i]]
            current_sum = data[i][1]

    clusters.append((current_cluster, current_sum))
    return clusters

symbolNumList =  ['14160', '42008487', '42003287']
symbolNameList = ['ES', 'NQ', 'YM']


intList = [str(i) for i in range(3,70)]

#vaildClust = [str(i) for i in range(0,200)]

#vaildTPO = [str(i) for i in range(1,500)]

#covarianceList = [str(round(i, 2)) for i in [x * 0.01 for x in range(1, 1000)]]

#gclient = storage.Client(project="stockapp-401615")
#bucket = gclient.get_bucket("stockapp-storage")

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = "stockapp-storage"
bucket = client.bucket(bucket_name)

import re
ftName = 'NQ' 
# Initialize Google Cloud Storage client
gclient = storage.Client()
bucket_name = "stockapp-storage"
prefix = ftName+"OverallTopOrders/"  # Filter files in 'oldData/' folder containing "NQ"

# Get bucket reference
bucket = gclient.bucket(bucket_name)

# List all files in the 'oldData/' directory containing "NQ"
blobs = list(bucket.list_blobs(prefix=prefix))

# Regex to extract the first date occurrence (_YYYY-MM-DD_)
date_pattern = re.compile(r"_(\d{4}-\d{2}-\d{2})_")

# Function to extract the first date occurrence
def extract_first_date(filename):
    match = date_pattern.search(filename)  # Find the first match
    return match.group(1) if match else "0000-00-00"  # Default for sorting safety

# Filter & sort files by the first extracted date
filtered_files = sorted(
    [blob.name for blob in blobs if ftName in blob.name],
    key=lambda x: extract_first_date(x),  # Sort by first date match
    reverse=False  # Sort from latest to oldest
)

#import duckdb
#from google.api_core.exceptions import NotFound
from dash import Dash, dcc, html, Input, Output, callback, State, callback_context
initial_inter = 1800000  # Initial interval #210000#250000#80001
subsequent_inter = 300000  # Subsequent interval
app = Dash()
app.title = "EnVisage"
app.layout = html.Div([
    
    dcc.Graph(id='graph', config={'modeBarButtonsToAdd': ['drawline']}),
    dcc.Interval(
        id='interval',
        interval=initial_inter,
        n_intervals=0,
      ),

    

    html.Div([
        html.Div([
            dcc.Input(id='input-on-submit', type='text', className="input-field"),
            html.Button('Submit', id='submit-val', n_clicks=0, className="submit-button"),
            html.Div(id='container-button-basic', children="Enter a symbol from ES, NQ", className="label-text"),
        ], className="sub-container"),
        dcc.Store(id='stkName-value'),

        html.Div([
            dcc.Input(id='input-on-interv', type='text', className="input-field"),
            html.Button('Submit', id='submit-interv', n_clicks=0, className="submit-button"),
            html.Div(id='interv-button-basic', children="Enter interval from 3-30, Default 10 mins", className="label-text"),
        ], className="sub-container"),
        dcc.Store(id='interv-value'),
    ], className="main-container"),

    dcc.Store(id='data-store'),
    dcc.Store(id='previous-interv'),
    dcc.Store(id='previous-stkName'),
    dcc.Store(id='interval-time', data=initial_inter),
    dcc.Store(id='graph-layout'),
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
        return 'The input symbol '+str(value)+" is not accepted please try different symbol from  |'ES', 'NQ'|", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(
    Output('interv-value', 'data'),
    Output('interv-button-basic', 'children'),
    Input('submit-interv', 'n_clicks'),
    State('input-on-interv', 'value'),
    prevent_initial_call=True
)
def update_interval(n_clicks, value):
    value = str(value)
    
    if value in intList:
        print('The input interval was "{}" '.format(value))
        return str(value), str(value), 
    else:
        return 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|", 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|"



@callback(
    [Output('data-store', 'data'),
        Output('graph', 'figure'),
        Output('previous-stkName', 'data'),
        Output('previous-interv', 'data'),
        Output('interval', 'interval'),
        Output('graph-layout', 'data')],
    [Input('interval', 'n_intervals'),
     Input('graph', 'relayoutData')], 
    [State('stkName-value', 'data'),
        State('interv-value', 'data'),
        State('data-store', 'data'),
        State('previous-stkName', 'data'),
        State('previous-interv', 'data'),
        State('interval-time', 'data'),
        State('graph-layout', 'data')],
    prevent_initial_call=False
)

def update_graph_live(n_intervals, relayout_data, sname, interv, stored_data, previous_stkName, previous_interv, interval_time, layout_data): #interv
    
    #print(sname, interv, stored_data, previous_stkName)
    #print(interv)

    if sname in symbolNameList:
        stkName = sname
        symbolNum = symbolNumList[symbolNameList.index(stkName)]   
    else:
        stkName = 'NQ' 
        sname = 'NQ'
        symbolNum = symbolNumList[symbolNameList.index(stkName)]
        
    if interv not in intList:
        interv = '60'
        
    #clustNum = '20'
        
    #tpoNum = '500'


    
        
    if sname != previous_stkName or interv != previous_interv:
        stored_data = None
    
        
    ctime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    if True:#stored_data is None:
        print('Newstored')
        blob = bucket.blob('Daily'+stkName)

        # Download CSV content as string
        csv_data = blob.download_as_text()
    
        # Convert to DataFrame using StringIO
        prevDf = pd.read_csv(StringIO(csv_data))
        
        #stored_data = {'df': prevDf.values.tolist()}

        with ThreadPoolExecutor(max_workers=2) as executor:
            #if sname != previous_stkName:
            # Download everything when stock name changes
            futures = [
                executor.submit(download_data, bucket, 'FuturesOHLC' + str(symbolNum)),
                executor.submit(download_data, bucket, 'FuturesTrades' + str(symbolNum)),]
                #executor.submit(download_daily_data, bucket, stkName)]
            
            FuturesOHLC, FuturesTrades = [future.result() for future in futures] #, prevDf


        # Process data with pandas directly
        FuturesOHLC = pd.read_csv(io.StringIO(FuturesOHLC), header=None)
        FuturesTrades = pd.read_csv(io.StringIO(FuturesTrades), header=None)
        
                
                
        aggs = [ ] 
        for row in FuturesOHLC.itertuples(index=False):
            # Extract values from the row, where row[0] corresponds to the first column, row[1] to the second, etc.
            hourss = datetime.fromtimestamp(int(row[0]) // 1000000000).hour
            hourss = f"{hourss:02d}"  # Ensure it's a two-digit string
            minss = datetime.fromtimestamp(int(row[0]) // 1000000000).minute
            minss = f"{minss:02d}"  # Ensure it's a two-digit string
            
            # Construct the time string
            opttimeStamp = f"{hourss}:{minss}:00"
            
            # Append the transformed row data to the aggs list
            aggs.append([
                row[2] / 1e9,  # Convert the value at the third column (open)
                row[3] / 1e9,  # Convert the value at the fourth column (high)
                row[4] / 1e9,  # Convert the value at the fifth column (low)
                row[5] / 1e9,  # Convert the value at the sixth column (close)
                int(row[6]),   # Volume
                opttimeStamp,  # The formatted timestamp
                int(row[0]),   # Original timestamp
                int(row[1])    # Additional identifier or name
            ])
        
        df2 = pd.DataFrame(aggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',])
            
        df2['strTime'] = df2['timestamp'].apply(lambda x: pd.Timestamp(int(x) // 10**9, unit='s', tz='EST') )
        
        df2.set_index('strTime', inplace=True)
        df2['volume'] = pd.to_numeric(df2['volume'], downcast='integer')
        df_resampled2 = df2.resample(interv+'min').agg({
            'timestamp': 'first',
            'name': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'time': 'first',
            'volume': 'sum'
        })
        
        df_resampled2.reset_index(drop=True, inplace=True)
        df_resampled2.insert(0, "index_count", df_resampled2.index)
        df_resampled2.dropna(inplace=True)
        df_resampled2.reset_index(drop=True, inplace=True)
        
        timestamps = FuturesTrades.iloc[:, 0].values
            
        # Convert timestamps and extract hours and minutes vectorized
        seconds_timestamps = timestamps // 1000000000
        dt_array = np.array([datetime.fromtimestamp(ts) for ts in seconds_timestamps])

        # Format hours and minutes
        hours = np.array([f"{dt.hour:02d}" for dt in dt_array])
        minutes = np.array([f"{dt.minute:02d}" for dt in dt_array])

        # Create formatted timestamps
        opt_timestamps = np.array([f"{h}:{m}:00" for h, m in zip(hours, minutes)])

        # Create indices array
        indices = np.arange(len(timestamps))

        # Create the AllTrades array efficiently
        AllTrades = np.column_stack([
            FuturesTrades.iloc[:, 1].values / 1e9,  # Scale by 1e9
            FuturesTrades.iloc[:, 2].values,
            timestamps,
            np.zeros(len(timestamps), dtype=int),
            indices,
            FuturesTrades.iloc[:, 3].values,
            opt_timestamps
        ])

           

        #combined_trades = pd.concat([pd.DataFrame(stored_data['trades'], columns=['0','1','2','3','4','5','6']), pd.DataFrame(AllTrades, columns=['0','1','2','3','4','5','6'])], ignore_index=True)


        dtimeEpoch_np = np.array(df_resampled2['timestamp'].dropna().values)
        dtime_np = np.array(df_resampled2['time'].dropna().values)
        tradeEpoch_np = np.array([i[2] for i in AllTrades])  # Extract timestamp from trades

        # Find the nearest tradeEpoch index using NumPy vectorization
        indices = np.searchsorted(tradeEpoch_np, dtimeEpoch_np, side='left')

        # Create `make` list using NumPy
        make = np.column_stack((dtimeEpoch_np, dtime_np, indices)).tolist()

        # Faster dictionary initialization using dictionary comprehension
        timeDict = {dtime_np[i]: [0, 0, 0] for i in range(len(dtime_np))}

        # Initialize troPerCandle and footPrint as empty lists
        troPerCandle = []
        #footPrint = []

        all_trades_np = np.array(AllTrades, dtype=object)


        #trade_prices = all_trades_np[:, 0].astype(float)  # Convert to float for numerical operations
        #trade_qty = all_trades_np[:, 1].astype(int)
        #trade_types = all_trades_np[:, 5] 


        for tr in range(len(make)):
            start_idx = make[tr][2]
            end_idx = make[tr+1][2] if tr+1 < len(make) else len(AllTrades)

            # Get trades for this time window
            tempList = all_trades_np[start_idx:end_idx]

            # Extract prices for binning
            if len(tempList) > 0:

                # Store top 100 largest trades (fast NumPy sorting)
                sorted_trades = tempList[np.argsort(tempList[:, 1].astype(int))][-100:].tolist()
                troPerCandle.append([make[tr][1], sorted_trades])

                # Aggregate buy/sell/neutral trade volumes
                for row in tempList:
                    if row[5] == "B":
                        timeDict[make[tr][1]][0] += row[1]
                    elif row[5] == "A":
                        timeDict[make[tr][1]][1] += row[1]
                    elif row[5] == "N":
                        timeDict[make[tr][1]][2] += row[1]
                        
        timeDict_np = np.array(list(timeDict.values()))
        sums = timeDict_np.sum(axis=1)
        ratios = np.divide(timeDict_np, sums[:, None], where=sums[:, None] != 0)  # Avoid division by zero

        timeFrame = [[timee, ""] + timeDict[timee] + ratios[i].tolist() for i, timee in enumerate(timeDict)]  
        
        timestamps = df_resampled2['timestamp'].values
        times = df_resampled2['time'].values
        
        valist = []
        for it in range(len(make)):
            # Slice AllTrades efficiently
            if it+1 < len(make):
                tempList = all_trades_np[:make[it+1][2]]  # Faster slicing
            else:
                tempList = all_trades_np
                
            df_slice = df_resampled2.iloc[:it+1]  # Faster than df[:it+1]
            temphs = historV1(df_slice, 500, {}, tempList.tolist(), [])
            vA = valueAreaV3(temphs[0])
            valist.append(vA + [timestamps[it], times[it], temphs[2]])
            
        
        top100perCandle = []
        for it in range(1, len(make)):  # Start from 1 to allow it-1 access
            start_idx = make[0][2]  # Always start from the beginning of the day's trades
            end_idx = make[it][2]   # Up to current candle
        
            # Get trades in the window
            trades_in_window = all_trades_np[start_idx:end_idx]
        
            # Get top 200 trades by quantity
            top_trades = trades_in_window[np.argsort(trades_in_window[:, 1].astype(int))][-200:].tolist()
        
            # Filter trades for the current candle interval
            lower_bound = make[it - 1][0]
            upper_bound = make[it][0]
            filtered_orders = [order for order in top_trades if lower_bound <= order[2] <= upper_bound]
        
            # Sum order quantities by side
            side_sums = defaultdict(float)
            for order in filtered_orders:
                side = order[5]
                side_sums[side] += order[1]
        
            # Append summary for current candle
            top100perCandle.append([
                make[it - 1][1],  # Time label
                side_sums.get('B', 0),
                side_sums.get('A', 0),
                side_sums.get('B', 0) - side_sums.get('A', 0)
            ])
            
        
        final_start = make[-1][0]
        final_time_label = make[-1][1]
        
        # Use all trades from the beginning of the day
        trades_in_window = all_trades_np[0:]
        top_trades = trades_in_window[np.argsort(trades_in_window[:, 1].astype(int))][-200:].tolist()
        
        # Only filter for trades **after the final_start**
        filtered_orders = [order for order in top_trades if order[2] >= final_start]
                
        '''
        trades_in_window = all_trades_np[0:]
        lower_bound = make[it][0]
        filtered_orders = [order for order in top_trades if order[2] >= lower_bound]
        '''
        side_sums = defaultdict(float)
        for order in filtered_orders:
            side = order[5]
            side_sums[side] += order[1]
        
    
        top100perCandle.append([
            final_time_label,
            side_sums.get('B', 0),
            side_sums.get('A', 0),
            side_sums.get('B', 0) - side_sums.get('A', 0)
        ])
        
        #stored_data = {'df': prevDf.values.tolist()}
        
        
        df_resampled2['topOrderOverallBuyInCandle'] = [i[1] for i in top100perCandle]
        df_resampled2['topOrderOverallSellInCandle'] = [i[2] for i in top100perCandle]
        df_resampled2['topDiffOverallInCandle']  = [i[3] for i in top100perCandle]
            
        df_resampled2['dailyLowVA'] = pd.Series([i[0] for i in valist])
        df_resampled2['dailyHighVA'] = pd.Series([i[1] for i in valist])
        df_resampled2['dailyPOC']  = pd.Series([i[2] for i in valist])
        df_resampled2['dailyPOC2']  = pd.Series([i[5] for i in valist])


        topBuys = []
        topSells = []

        # Iterate through troPerCandle and compute values
        for i in troPerCandle:
            tobuyss = sum(x[1] for x in i[1] if x[5] == 'B')  # Sum buy orders
            tosellss = sum(x[1] for x in i[1] if x[5] == 'A')  # Sum sell orders
            
            topBuys.append(tobuyss)  # Store buy values
            topSells.append(tosellss)  # Store sell values
              

        df_resampled2['topBuys'] = topBuys
        df_resampled2['topSells'] = topSells
        df_resampled2['topDiff'] = df_resampled2['topBuys'] - df_resampled2['topSells']
        df_resampled2['topDiffNega'] = ((df_resampled2['topBuys'] - df_resampled2['topSells']).apply(lambda x: x if x < 0 else np.nan)).abs()
        df_resampled2['topDiffPost'] = (df_resampled2['topBuys'] - df_resampled2['topSells']).apply(lambda x: x if x > 0 else np.nan)

        df_resampled2['percentile_topBuys'] =  [percentileofscore(df_resampled2['topBuys'][:i+1], df_resampled2['topBuys'][i], kind='mean') for i in range(len(df_resampled2))]
        df_resampled2['percentile_topSells'] = [percentileofscore(df_resampled2['topSells'][:i+1], df_resampled2['topSells'][i], kind='mean') for i in range(len(df_resampled2))] 

        df_resampled2['percentile_Posdiff'] =  [percentileofscore(df_resampled2['topDiffPost'][:i+1].dropna(), df_resampled2['topDiffPost'][i], kind='mean') if not np.isnan(df_resampled2['topDiffPost'][i]) else None for i in range(len(df_resampled2))]
        df_resampled2['percentile_Negdiff'] =  [percentileofscore(df_resampled2['topDiffNega'][:i+1].dropna(), df_resampled2['topDiffNega'][i], kind='mean') if not np.isnan(df_resampled2['topDiffNega'][i]) else None for i in range(len(df_resampled2))]

        df_resampled2['allDiff'] = [i[2]-i[3] for i in timeFrame]
        df_resampled2['buys'] = [i[2] for i in timeFrame]
        df_resampled2['sells'] = [i[3] for i in timeFrame]
        
        
        
        df_resampled2['datetime'] = pd.to_datetime(df_resampled2['timestamp'], unit='ns')

        # Convert to Eastern Time (automatically handles EST/EDT)
        df_resampled2['datetime_est'] = df_resampled2['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

        # Format as MM/DD/YYYY HH:MM in Eastern Time
        df_resampled2['formatted_date'] = df_resampled2['datetime_est'].dt.strftime('%m/%d/%Y %H:%M')

        df_resampled2['buyPercent'] = df_resampled2['buys'] / (df_resampled2['buys']+df_resampled2['sells'])
        df_resampled2['sellPercent'] = df_resampled2['sells'] / (df_resampled2['buys']+df_resampled2['sells'])

        df_resampled2['topBuysPercent'] = ((df_resampled2['topBuys']) / (df_resampled2['topBuys']+df_resampled2['topSells']))
        df_resampled2['topSellsPercent'] = ((df_resampled2['topSells']) / (df_resampled2['topBuys']+df_resampled2['topSells']))
        

        
        dtime = df_resampled2['time'].dropna().values.tolist()
        dtimeEpoch = df_resampled2['timestamp'].dropna().values.tolist()
        tradeEpoch = AllTrades[:, 2].tolist() #AllTrades.iloc[:, 2].tolist()

        alltimeDict = {}
        allmake = []
        for ttm in range(len(dtimeEpoch)):
            
            allmake.append([dtimeEpoch[ttm],dtime[ttm],bisect.bisect_left(tradeEpoch, dtimeEpoch[ttm])]) #min(range(len(tradeEpoch)), key=lambda i: abs(tradeEpoch[i] - dtimeEpoch[ttm]))
            alltimeDict[dtime[ttm]] = [0,0,0]
        
        blob = bucket.blob('Daily'+stkName+'lastVP')
        
        # Download the blob content as text
        blob_text = blob.download_as_text()
        
        # Split the text into a list (assuming each line is an item)
        lastVp = blob_text.splitlines()
        
        lastVp = [
            [float(x.strip()) for x in line.split(',')]
            for line in lastVp
        ]
        
        
            
        allvalist =[]
        prevHist = []
        for it in range(len(allmake)):
            if it+1 < len(allmake):
                tempList = AllTrades[allmake[it][2]:allmake[it+1][2]]
                if it == 0:
                    temphs = historV2(df_resampled2[:it+1],100,{}, tempList, [])
                    cHist = combine_histogram_data_2([lastVp,0], temphs)
                    prevHist = cHist
                    vA = valueAreaV3(cHist[0])
                    allvalist.append(vA  + [df_resampled2['timestamp'][it], df_resampled2['time'][it], cHist[1]]) 
                else:
                    temphs = historV2(df_resampled2[:it+1],100,{}, tempList, [])
                    cHist = combine_histogram_data_2(prevHist, temphs)
                    prevHist = cHist
                    vA = valueAreaV3(cHist[0])
                    allvalist.append(vA  + [df_resampled2['timestamp'][it], df_resampled2['time'][it], cHist[1]]) 
            else:
                tempList = AllTrades
                temphs = historV2(df_resampled2[:it+1],100,{}, tempList, [])
                if len(prevHist) == 0:
                    cHist = combine_histogram_data_2([lastVp,0], temphs)
                else:
                    cHist = combine_histogram_data_2(prevHist, temphs)
                prevHist = cHist
                vA = valueAreaV3(cHist[0])
                allvalist.append(vA  + [df_resampled2['timestamp'][it], df_resampled2['time'][it], cHist[1]]) 
            
            
        
        stored_data = {'df': prevDf.values.tolist(), 'top100perCandle':top100perCandle, 'valist':valist, 'lastVp':cHist, 'allvalist':allvalist} 
        
        df_resampled2['allLowVA'] = pd.Series([i[0] for i in allvalist])
        df_resampled2['allHighVA'] = pd.Series([i[1] for i in allvalist])
        df_resampled2['allPOC']  = pd.Series([i[2] for i in allvalist])
        df_resampled2['allPOC2']  = pd.Series([i[5] for i in allvalist])
        

                
    df = pd.concat([prevDf, df_resampled2], ignore_index=True)
    
    #n_rows   = len(df)
    #start_i  = int(n_rows * 0.97)
    #df['1ema'] = df['close'].ewm(span=1, adjust=False).mean()
    # 2) slice from that position to the end
    #df = df.iloc[start_i:].reset_index(drop=True)
    
    
    df['alltopDiffNega'] = ((df['topBuys'] - df['topSells']).apply(lambda x: x if x < 0 else np.nan)).abs()
    df['alltopDiffPost'] = (df['topBuys'] - df['topSells']).apply(lambda x: x if x > 0 else np.nan)
    
    df['allpercentile_topBuys'] =  [percentileofscore(df['topBuys'][:i+1], df['topBuys'][i], kind='mean') for i in range(len(df))]
    df['allpercentile_topSells'] = [percentileofscore(df['topSells'][:i+1], df['topSells'][i], kind='mean') for i in range(len(df))] 

    df['allpercentile_Posdiff'] =  [percentileofscore(df['alltopDiffPost'][:i+1].dropna(), df['alltopDiffPost'][i], kind='mean') if not np.isnan(df['alltopDiffPost'][i]) else None for i in range(len(df))]
    df['allpercentile_Negdiff'] =  [percentileofscore(df['alltopDiffNega'][:i+1].dropna(), df['alltopDiffNega'][i], kind='mean') if not np.isnan(df['alltopDiffNega'][i]) else None for i in range(len(df))]
    
    
    
    df['smoothed_1ema'] = butter_lowpass_realtime(df["close"],cutoff=0.5, order=2)
    df['POCDistanceEMA'] = (df['smoothed_1ema'] - df['allPOC']) / df['allPOC'] * 100
    
    putCandImb =  df.index[
        (df['topBuys'] > df['topSells']) &
        (df['percentile_topBuys'] > 95) &
        (df['topBuysPercent'] >= 0.65)
    ].tolist()
    callCandImb = df.index[
        (df['topSells'] > df['topBuys']) &
        (df['percentile_topSells'] > 95) &
        (df['topSellsPercent'] >= 0.65)
    ].tolist()
    
    
    putCandImb_1 =  df.index[
        (df['topBuys'] > df['topSells']) &
        (df['allpercentile_topBuys'] > 95) &
        (df['topBuysPercent'] >= 0.65)
    ].tolist()
    callCandImb_1 = df.index[
        (df['topSells'] > df['topBuys']) &
        (df['allpercentile_topSells'] > 95) &
        (df['topSellsPercent'] >= 0.65)
    ].tolist()
        
        
    previous_stkName = sname
    previous_interv = interv
    
    formatted_dates = df['formatted_date'].tolist()
    top_buys = df['topBuysPercent'].tolist()
    top_sells = df['topSellsPercent'].tolist()
    top_buys_count = df['topBuys'].tolist()
    top_sells_count = df['topSells'].tolist()
    topOrderOverallBuyInCandle = df['topOrderOverallBuyInCandle'].tolist()
    topOrderOverallSellInCandle = df['topOrderOverallSellInCandle'].tolist()
    topDiffOverallInCandle = df['topDiffOverallInCandle'].tolist()
    
    # Zip the three lists together
    zipped = zip(formatted_dates, top_buys, top_sells, top_buys_count, top_sells_count, topOrderOverallBuyInCandle, topOrderOverallSellInCandle, topDiffOverallInCandle)
    
    # Create a list of strings
    list_of_strings = [
    f"{dates}<br> Buys: {buy_count} : {round(buy_percent, 2)}<br> Sells: {sell_count} : {round(sell_percent, 2)}<br>"
    f"<br>OverallTopOrders in Candle: <br>"
    f"Buys : ({topOrderOverallBuyInCandle})<br>"
    f"Sells : ({topOrderOverallSellInCandle})<br>"
    f"Diff : ({topDiffOverallInCandle})<br>"
    for dates, buy_percent, sell_percent, buy_count, sell_count, topOrderOverallBuyInCandle, topOrderOverallSellInCandle, topDiffOverallInCandle in zipped
    ]
    
    #print(list_of_strings)
        
    
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                            specs=[[{}, {}],
                                   [{}, {}]], #[{"colspan": 1}, {}] [{"colspan": 1},{},][{}, {}, ]'+ '<br>' +' ( Put:'+str(putDecHalf)+'('+str(NumPutHalf)+') | '+'Call:'+str(CallDecHalf)+'('+str(NumCallHalf)+') '
                             horizontal_spacing=0.00, vertical_spacing=0.00, # subplot_titles=(stkName +' '+ str(datetime.now().time()))' (Sell:'+str(putDec)+' ('+str(round(NumPut,2))+') | '+'Buy:'+str(CallDec)+' ('+str(round(NumCall,2))+') \n '+' (Sell:'+str(thputDec)+' ('+str(round(thNumPut,2))+') | '+'Buy:'+str(thCallDec)+' ('+str(round(thNumCall,2))+') \n '
                             column_widths=[0.90,0.10],row_width=[0.20, 0.80,] ) #, column_widths=[0.85,0.15], 62

        
    
    
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name="OHLC",
                                 hovertext=list_of_strings),
                  row=1, col=1)
    
    

    #fig.add_trace(go.Scatter(x=df.index, y=df['POC2'], mode='lines',name='POC', hovertext=df['time'].tolist(), marker_color='#0000FF'))
    #fig.add_trace(go.Scatter(x=df.index, y=df['LowVA'], mode='lines', opacity=0.3, name='LowVA', line=dict(color='purple')))
    #fig.add_trace(go.Scatter(x=df.index, y=df['HighVA'], mode='lines', opacity=0.3, name='HighVA', line=dict(color='purple')))

    #if (abs(df['allPOC'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allPOC'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 or (abs(df['allPOC'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allPOC'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 :
    fig.add_trace(go.Scatter(x=df.index, y=df['allPOC'], mode='lines',name='allPOC', hovertext=df['time'].tolist(), marker_color='#0000FF'))
    
    #if (abs(df['allPOC2'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allPOC2'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 or (abs(df['allPOC2'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allPOC2'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 :
    fig.add_trace(go.Scatter(x=df.index, y=df['allPOC2'], mode='lines',name='allPOC2', hovertext=df['time'].tolist(), marker_color='#0000FF'))
    
    #if (abs(df['allHighVA'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allHighVA'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 or (abs(df['allHighVA'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allHighVA'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 :
    fig.add_trace(go.Scatter(x=df.index, y=df['allHighVA'], mode='lines', opacity=0.3, name='allHighVA', line=dict(color='purple')))
    
    #if (abs(df['allLowVA'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allLowVA'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 or (abs(df['allLowVA'][len(df)-1] - df['1ema'][len(df)-1]) / ((df['allLowVA'][len(df)-1] + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 :
    fig.add_trace(go.Scatter(x=df.index, y=df['allLowVA'], mode='lines', opacity=0.3, name='allLowVA', line=dict(color='purple')))
    
    
    fig.add_trace(go.Scatter(x=df.index, y=df['dailyPOC'], mode='lines',name='dailyPOC', hovertext=df['time'].tolist(), marker_color='#16FF32'))
    fig.add_trace(go.Scatter(x=df.index, y=df['dailyPOC2'], mode='lines',name='dailyPOC2', hovertext=df['time'].tolist(), marker_color='#16FF32'))
        

    fig.add_trace(go.Candlestick(
        x=[df.index[i] for i in range(len(top_buys_count)) if top_buys_count[i] > top_sells_count[i]],
        open=[df['open'][i] for i in range(len(top_buys_count)) if top_buys_count[i] > top_sells_count[i]],
        high=[df['high'][i] for i in range(len(top_buys_count)) if top_buys_count[i] > top_sells_count[i]],
        low=[df['low'][i] for i in range(len(top_buys_count)) if top_buys_count[i] > top_sells_count[i]],
        close=[df['close'][i] for i in range(len(top_buys_count)) if top_buys_count[i] > top_sells_count[i]],
        increasing={'line': {'color': 'teal'}},
        decreasing={'line': {'color': 'teal'}},
        hovertext=[list_of_strings[i] for i in range(len(top_buys_count)) if top_buys_count[i] > top_sells_count[i]],
        hoverlabel=dict(
             bgcolor="teal",
             font=dict(color="white", size=10),
             ),
        name='' ),
    row=1, col=1)

        

    fig.add_trace(go.Candlestick(
        x=[df.index[i] for i in range(len(top_buys_count)) if top_buys_count[i] < top_sells_count[i]],
        open=[df['open'][i] for i in range(len(top_buys_count)) if top_buys_count[i] < top_sells_count[i]],
        high=[df['high'][i] for i in range(len(top_buys_count)) if top_buys_count[i] < top_sells_count[i]],
        low=[df['low'][i] for i in range(len(top_buys_count)) if top_buys_count[i] < top_sells_count[i]],
        close=[df['close'][i] for i in range(len(top_buys_count)) if top_buys_count[i] < top_sells_count[i]],
        increasing={'line': {'color': 'pink'}},
        decreasing={'line': {'color': 'pink'}},
        hovertext=[list_of_strings[i]  for i in range(len(top_buys_count)) if top_buys_count[i] < top_sells_count[i]],
        hoverlabel=dict(
             bgcolor="pink",
             font=dict(color="black", size=10),
             ),
        name='' ),
    row=1, col=1)
    '''
    last_ema = df['1ema'].iloc[-1]

    # 2) threshold in percent
    threshold = 4.25
    
    # 3) filter the bars in cHist[0]
    filtered = [i for i in cHist[0] if abs(i[0] - last_ema) / ((i[0] + last_ema) / 2) * 100 <= threshold or abs(i[3] - last_ema) / ((i[3] + last_ema) / 2) * 100 <= threshold]
    

    # 4) build x, y, hovertext from the filtered list (reversed if you like)
    x_vals      = [i[1] for i in filtered[::-1]]
    y_labels    = [i[0] for i in filtered[::-1]]
    hover_texts = [f"Edge: {i[0]} – {i[3]}" for i in filtered[::-1]]
    
    # 5) add the Bar trace
    fig.add_trace(
        go.Bar(
            x            = x_vals,
            y            = y_labels,
            orientation  = 'h',
            textposition = 'auto',
            marker_color = 'teal',
            hovertext    = hover_texts
        ),
        row=1, col=2
    )
    
    '''
    fig.add_trace(go.Bar(
        x=[i[1] for i in cHist[0][::-1]],  # bar length 
        y=[i[0] for i in cHist[0][::-1]],  # y-axis labels
        orientation='h',
        #text=[str(i[1]) for i in bbbb[1]],  # show index 1 as bar label
        textposition='auto',
        marker_color='teal',  # static color for now
        hovertext=[f"Edge: {i[0]} - {i[3]}" for i in cHist[0][::-1]]  # custom hover text
    ),
        row=1, col=2
    )
    


    if len(callCandImb) > 0:
        fig.add_trace(go.Candlestick(
            x=callCandImb,  # Directly use the index list
            open=df.loc[callCandImb, 'open'].values,  # Access using .loc[]
            high=df.loc[callCandImb, 'high'].values,
            low=df.loc[callCandImb, 'low'].values,
            close=df.loc[callCandImb, 'close'].values,
            increasing={'line': {'color': 'black'}},
            decreasing={'line': {'color': 'black'}},
            hovertext=[
                f"({df.loc[i, 'buys']}) {round(df.loc[i, 'buyPercent'], 2)} Bid "
                f"({df.loc[i, 'sells']}) {round(df.loc[i, 'sellPercent'], 2)} Ask <br>"
                f"{df.loc[i, 'allDiff']} <br>TopOrders: <br>"
                f"({df.loc[i, 'topBuys']}) {round(df.loc[i, 'topBuysPercent'], 2)} Bid "
                f"({df.loc[i, 'topSells']}) {round(df.loc[i, 'topSellsPercent'], 2)} Ask<br>"
                f"{df.loc[i, 'formatted_date']}"
                f"<br>OverallTopOrders in Candle: <br>"
                f"Buys : ({df.loc[i, 'topOrderOverallBuyInCandle']})<br>"
                f"Sells : ({df.loc[i,'topOrderOverallSellInCandle']})<br>"
                f"Diff : ({df.loc[i,'topDiffOverallInCandle']})<br>"
                
                
                for i in callCandImb
            ],
            hoverlabel=dict(
                 bgcolor="black",
                 font=dict(color="white", size=13),
                 ),
            name='Sellimbalance Today' ),
        row=1, col=1)

        
    if len(putCandImb) > 0:
        fig.add_trace(go.Candlestick(
            x=putCandImb,  # Directly use the index list
            open=df.loc[putCandImb, 'open'].values,  # Access using .loc[]
            high=df.loc[putCandImb, 'high'].values,
            low=df.loc[putCandImb, 'low'].values,
            close=df.loc[putCandImb, 'close'].values,
            increasing={'line': {'color': '#16FF32'}},
            decreasing={'line': {'color': '#16FF32'}},
            hovertext=[
                f"({df.loc[i, 'buys']}) {round(df.loc[i, 'buyPercent'], 2)} Bid "
                f"({df.loc[i, 'sells']}) {round(df.loc[i, 'sellPercent'], 2)} Ask <br>"
                f"{df.loc[i, 'allDiff']} <br>TopOrders: <br>"
                f"({df.loc[i, 'topBuys']}) {round(df.loc[i, 'topBuysPercent'], 2)} Bid "
                f"({df.loc[i, 'topSells']}) {round(df.loc[i, 'topSellsPercent'], 2)} Ask<br>"
                f"{df.loc[i, 'formatted_date']}"
                f"<br>OverallTopOrders in Candle: <br>"
                f"Buys : ({df.loc[i, 'topOrderOverallBuyInCandle']})<br>"
                f"Sells : ({df.loc[i,'topOrderOverallSellInCandle']})<br>"
                f"Diff : ({df.loc[i,'topDiffOverallInCandle']})<br>"
                for i in putCandImb
            ],
            hoverlabel=dict(
                bgcolor="#2CA02C",
                font=dict(color="white", size=13),
            ),
            name='BuyImbalance Today'
        ), row=1, col=1)
        
        
    
    if len(callCandImb_1) > 0:
        fig.add_trace(go.Candlestick(
            x=callCandImb_1,  # Directly use the index list
            open=df.loc[callCandImb_1, 'open'].values,  # Access using .loc[]
            high=df.loc[callCandImb_1, 'high'].values,
            low=df.loc[callCandImb_1, 'low'].values,
            close=df.loc[callCandImb_1, 'close'].values,
            increasing={'line': {'color': 'crimson'}},
            decreasing={'line': {'color': 'crimson'}},
            hovertext=[
                f"({df.loc[i, 'buys']}) {round(df.loc[i, 'buyPercent'], 2)} Bid "
                f"({df.loc[i, 'sells']}) {round(df.loc[i, 'sellPercent'], 2)} Ask <br>"
                f"{df.loc[i, 'allDiff']} <br>TopOrders: <br>"
                f"({df.loc[i, 'topBuys']}) {round(df.loc[i, 'topBuysPercent'], 2)} Bid "
                f"({df.loc[i, 'topSells']}) {round(df.loc[i, 'topSellsPercent'], 2)} Ask<br>"
                f"{df.loc[i, 'formatted_date']}"
                f"<br>OverallTopOrders in Candle: <br>"
                f"Buys : ({df.loc[i, 'topOrderOverallBuyInCandle']})<br>"
                f"Sells : ({df.loc[i,'topOrderOverallSellInCandle']})<br>"
                f"Diff : ({df.loc[i,'topDiffOverallInCandle']})<br>"
                for i in callCandImb_1
            ],
            hoverlabel=dict(
                 bgcolor="crimson",
                 font=dict(color="white", size=13),
                 ),
            name='AllSellimbalance Overall' ),
        row=1, col=1)

        
    if len(putCandImb_1) > 0:
        fig.add_trace(go.Candlestick(
            x=putCandImb_1,  # Directly use the index list
            open=df.loc[putCandImb_1, 'open'].values,  # Access using .loc[]
            high=df.loc[putCandImb_1, 'high'].values,
            low=df.loc[putCandImb_1, 'low'].values,
            close=df.loc[putCandImb_1, 'close'].values,
            increasing={'line': {'color': '#2ED9FF'}},
            decreasing={'line': {'color': '#2ED9FF'}},
            hovertext=[
                f"({df.loc[i, 'buys']}) {round(df.loc[i, 'buyPercent'], 2)} Bid "
                f"({df.loc[i, 'sells']}) {round(df.loc[i, 'sellPercent'], 2)} Ask <br>"
                f"{df.loc[i, 'allDiff']} <br>TopOrders: <br>"
                f"({df.loc[i, 'topBuys']}) {round(df.loc[i, 'topBuysPercent'], 2)} Bid "
                f"({df.loc[i, 'topSells']}) {round(df.loc[i, 'topSellsPercent'], 2)} Ask<br>"
                f"{df.loc[i, 'formatted_date']}"
                f"<br>OverallTopOrders in Candle: <br>"
                f"Buys : ({df.loc[i, 'topOrderOverallBuyInCandle']})<br>"
                f"Sells : ({df.loc[i,'topOrderOverallSellInCandle']})<br>"
                f"Diff : ({df.loc[i,'topDiffOverallInCandle']})<br>"
                for i in putCandImb_1
            ],
            hoverlabel=dict(
                bgcolor="#2ED9FF",
                font=dict(color="white", size=13),
            ),
            name='AllBuyImbalance Overall'
        ), row=1, col=1)
        
        
    colors = ['maroon']
    for val in range(1,len(df['topDiffOverallInCandle'])):
        if df['topDiffOverallInCandle'][val] > 0:
            color = 'teal'
            if df['topDiffOverallInCandle'][val] > df['topDiffOverallInCandle'][val-1]:
                color = '#54C4C1' 
        else:
            color = 'maroon'
            if df['topDiffOverallInCandle'][val] < df['topDiffOverallInCandle'][val-1]:
                color='crimson' 
        colors.append(color)
    fig.add_trace(go.Bar(x=df.index, y=df['topDiffOverallInCandle'], marker_color=colors), row=2, col=1)
     
    
    
    for i in range(8):
        col = f"tp100allDay-{i}"
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col
            )
        )


    '''
    if layout_data:
        if 'xaxis.range[0]' in layout_data and 'xaxis.range[1]' in layout_data:
            fig.update_layout(xaxis_range=[layout_data['xaxis.range[0]'], layout_data['xaxis.range[1]']])
        if 'yaxis.range[0]' in layout_data and 'yaxis.range[1]' in layout_data:
            fig.update_layout(yaxis_range=[layout_data['yaxis.range[0]'], layout_data['yaxis.range[1]']])
    #if 'POCDistanceEMA' in df.columns:
    
    colors = ['maroon']

    for val in range(1, len(df['POCDistanceEMA'])):
        if df['POCDistanceEMA'].iloc[val] > 0:
            color = 'teal'
            if df['POCDistanceEMA'].iloc[val] > df['POCDistanceEMA'].iloc[val-1]:
                color = '#54C4C1' 
        else:
            color = 'maroon'
            if df['POCDistanceEMA'].iloc[val] < df['POCDistanceEMA'].iloc[val-1]:
                color = 'crimson'
        colors.append(color)
    
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['POCDistanceEMA'], 
            marker_color=colors
        ),
        row=2, col=1
    )
    '''
    '''        
    blob = bucket.blob('Daily'+stkName+'topOrders')
    
    # Download the blob content as text
    blob_text = blob.download_as_text()
    
    # Split the text into a list (assuming each line is an item)
    dailyNQtopOrders = blob_text.splitlines()
    
        # Step 1: Split each line into fields
    split_data = [row.split(', ') for row in dailyNQtopOrders]
    
    # Step 2: Convert numeric fields properly
    converted_data = []
    for row in split_data:
        new_row = [
            float(row[0]),       # price -> float
            int(row[1]),         # quantity -> int
            int(row[2]),         # id -> int
            int(row[3]),         # field4 -> int
            int(row[4]),         # field5 -> int
            row[5],              # letter -> str
            row[6]               # time -> str
        ]
        converted_data.append(new_row)
    
    # Step 3: Make it a numpy array
    array_data = np.array(converted_data, dtype=object)
    
    combined_trades = np.concatenate((array_data, AllTrades), axis=0)
    combined_trades = pd.DataFrame(combined_trades)
    
    combined_trades_sorted = combined_trades.sort_values(by=combined_trades.columns[1], ascending=False)
    combined_trades_sorted = combined_trades_sorted.iloc[:1000]
    #prices = combined_trades_sorted.iloc[:, 0,1].sort_values().tolist()  # Sorted list of prices
    prices = combined_trades_sorted.iloc[:, [0, 1, 5]].sort_values(by=combined_trades_sorted.columns[0]).values.tolist()

    
    differences = [abs(prices[i + 1][0] - prices[i][0]) for i in range(len(prices) - 1)]
    average_difference = sum(differences) / len(differences)

    # Step 3: Find clusters
    cdata = find_clusters_1(prices, average_difference)

    
    #mazz = sum(cluster[1] for cluster in cdata) / len(cdata)
    volumes = [cluster[1] for cluster in cdata]
    mazz = np.percentile(volumes, 70) 
    max_volume = max(cluster[1] for cluster in cdata if cluster[1] > mazz)
    
    
    for cluster in cdata:
        if cluster[1] > mazz:
            #for i in cluster[0]:
            maxNum = max([i[0] for i in cluster[0]])
            minNum = min([i[0] for i in cluster[0]]) 
            bidCount = sum([i[1] for i in cluster[0] if i[2] == 'B'])
            askCount = sum([i[1] for i in cluster[0] if i[2] == 'A'])
            totalVolume = bidCount + askCount
            if totalVolume > 0:
                askDec = round(askCount / totalVolume, 2)
                bidDec = round(bidCount / totalVolume, 2)
            else:
                askDec = bidDec = 0

            opac = round(cluster[1] / max_volume,3)
            #if (abs(float(maxNum) - df['1ema'][len(df)-1]) / ((float(maxNum) + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 or (abs(float(minNum) - df['1ema'][len(df)-1]) / ((float(minNum) + df['1ema'][len(df)-1]) / 2)) * 100 <= 3.25 :
            fillcolor = (
                "crimson" if askCount > bidCount else
                "teal" if bidCount > askCount else
                "gray"
            )
            linecolor = f'rgba(220,20,60,{opac})' if askCount > bidCount else (
                        f'rgba(0,139,139,{opac})' if bidCount > askCount else 'gray')
                
            fig.add_shape(
                type="rect",
                y0=minNum, y1=maxNum, x0=-1, x1=len(df),
                fillcolor=fillcolor,
                opacity=opac#round(cluster[1] / max_volume,3)
            )
                
            # Upper line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[maxNum] * len(df),
                line_color=linecolor,#f"rgba(128, 128, 128, {round(cluster[1] / max_volume,3)})",
                text=f"{maxNum} : {cluster[1]}",
                textposition="bottom left",
                name=f"{maxNum} : {cluster[1]}",
                showlegend=False,
                mode='lines'
            ), row=1, col=1)

            # Lower line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[minNum] * len(df),
                line_color=linecolor,#f"rgba(128, 128, 128, {round(cluster[1] / max_volume,3)})",
                text=f"{minNum} : {cluster[1]}",
                textposition="bottom left",
                name=f"{minNum} : {cluster[1]}",
                showlegend=False,
                mode='lines'
            ), row=1, col=1)
     
    
    
    #mazz = max(len(cluster) for cluster in cdata)
    clustercount = 5

    for cluster in cdata:
        if len(cluster) >= clustercount:
            bidCount = 0
            askCount = 0
            strCount = ''
            
            price_low = min(cluster)
            price_high = max(cluster)
            
            

            for _, row in combined_trades_sorted.iterrows():
                price = row.iloc[0]
                volume = row.iloc[1]
                side = row.iloc[5]

                if price_low <= price <= price_high:#cluster[-1] <= price <= cluster[0]:  # price within cluster range
                    if side == 'B':
                        bidCount += volume
                    elif side == 'A':
                        askCount += volume
                    strCount += str(price) + " " + str(volume) + " " + str(side) + "<br>"

            totalVolume = bidCount + askCount
            if totalVolume > 0:
                askDec = round(askCount / totalVolume, 2)
                bidDec = round(bidCount / totalVolume, 2)
            else:
                askDec = bidDec = 0

            opac = round((len(cluster) / mazz) / 2, 2)
            fillcolor = (
                "crimson" if askCount > bidCount else
                "teal" if bidCount > askCount else
                "gray"
            )
            linecolor = f'rgba(220,20,60,{opac})' if askCount > bidCount else (
                        f'rgba(0,139,139,{opac})' if bidCount > askCount else 'gray')

            # Rectangle for cluster zone
            fig.add_shape(
                type="rect",
                y0=cluster[0], y1=cluster[-1], x0=-1, x1=len(df),
                fillcolor=fillcolor,
                opacity=opac
            )

            # Upper line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[cluster[0]] * len(df),
                line_color=linecolor,
                text=f"{cluster[0]} ({totalVolume}) ({len(cluster)}) Ask:({askDec}) {askCount} | Bid: ({bidDec}) {bidCount} <br> {strCount}",
                textposition="bottom left",
                name=f"{cluster[0]} ({totalVolume}) ({len(cluster)}) Ask:({askDec}) {askCount} | Bid: ({bidDec}) {bidCount}",
                showlegend=False,
                mode='lines'
            ), row=1, col=1)

            # Lower line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[cluster[-1]] * len(df),
                line_color=linecolor,
                text=f"{cluster[-1]} ({totalVolume}) ({len(cluster)}) Ask:({askDec}) {askCount} | Bid: ({bidDec}) {bidCount} <br> {strCount}",
                textposition="bottom left",
                name=f"{cluster[-1]} ({totalVolume}) ({len(cluster)}) Ask:({askDec}) {askCount} | Bid: ({bidDec}) {bidCount}",
                showlegend=False,
                mode='lines'
            ), row=1, col=1)
            
            
    '''
            
    fig.update_layout(title=ctime,
                          paper_bgcolor='#E5ECF6',
                          showlegend=False,
                          height=880,
                          xaxis_rangeslider_visible=False,)    

    fig.update_xaxes(autorange="reversed", row=1, col=2) 
    fig.update_xaxes(showticklabels=False, row=2, col=1)   

    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'graph.relayoutData':
        # Only update the layout_data when the user interacts with the graph
        if relayout_data and ('xaxis.range[0]' in relayout_data or 'yaxis.range[0]' in relayout_data):
            layout_data = relayout_data
    
    # Apply stored layout to new figure
    if layout_data:
        # Apply x-axis range if available
        if 'xaxis.range[0]' in layout_data and 'xaxis.range[1]' in layout_data:
            fig.update_layout(xaxis_range=[layout_data['xaxis.range[0]'], layout_data['xaxis.range[1]']])
        
        # Apply y-axis range if available
        if 'yaxis.range[0]' in layout_data and 'yaxis.range[1]' in layout_data:
            fig.update_layout(yaxis_range=[layout_data['yaxis.range[0]'], layout_data['yaxis.range[1]']])
            
    #fig.show(config={'modeBarButtonsToAdd': ['drawline']})
    

           
    if interval_time == initial_inter:
        interval_time = subsequent_inter
        
    if stkName != previous_stkName  or interv != previous_interv:
        interval_time = initial_inter


    return stored_data, fig, previous_stkName, previous_interv, interval_time, relayout_data
        
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)  
 