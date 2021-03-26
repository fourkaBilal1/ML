# Report Project Mines ML 2020


**Author**: **Bilal Fourka**

Link to the github repository : https://github.com/fourkaBilal1/MinesProject2020


## Introduction

**Urban traffic counts** are used mainly by municipal services dependent on the State, to **size** the traffic lanes as best as possible, especially in **their width and the choice of materials**. The main aim is to "**make profitable**" road and street development projects, so that they are as cheap as possible in their establishment and maintenance, by adapting them to the ***volume and nature of the flows*** that travel them. Although they are as important for the **modernization of streets as roads**, urban counts face difficulties in applying techniques designed for the road: urban travel is more difficult to grasp in its geography and diversity (modes, patterns). **Counts** can be **manual** (with field agents) or **automatic** (wires on the ground that transmit information to a box located on the edge of the track or in teletransmission), or even **estimated** (Household-Travel Surveys). You can have **counts by sections** or **counts ropes** "origin-destination" (on a route around a city for example), generalized since the 1953.

It  is  for  this  motivation  that  we  will  work  on  the  traffic  data  collected  from  several  Wavetronix  radar  sensors  deployed  by  the  city  of  Austin  in  the  United  States  of  America. The  dataset  is  augmented with geo coordinates from sensor location dataset.

## Data


Our data [Radar Traffic Data](https://www.kaggle.com/vinayshanbhag/radar-traffic-data) is hosted on Kaggle. It is provided by [Vinay Shanbhag](https://www.kaggle.com/vinayshanbhag).

Our  goal  is  to  predict  the  number  of  future  cars  that  will  pass  a  road  based  on  historical  data  from  cars  counted  in  the  city  of  Austin. Let’s download the data and load it into a Pandas data frame:

```
df = pd.read_csv("../Radar_Traffic_Counts.csv")
df.head()
```
Out[40]:

| |location_name| location_latitude|location_longitude  |Year  |Month|Day  | Day of Week  |Hour  |Minute 	|Time Bin|Direction  		|Volume |
|-|------------|---------|----------|----|--|--|-|--------|-|-|-----------|---|
|0|2021 BLK... |30.248691|-97.770409|2018|1 |23|2|22|15|22:15|None|4|
|1|CAPITAL ... |30.371674|-97.785660|2017|12|16|6|19|45|19:45|NB|103|
|2|400 BLK ... |30.264245|-97.765802|2018|1 |23|2|21|45|21:45|SB|44|
|3|400 BLK ... |30.264245|-97.765802|2018|1 |23|2|21|45|21:45|NB|13|
|4|2021 BLK... |30.248691|-97.770409|2018|1 |23|2|22|15|22:15|None|0|

[5 rows x 12 columns]


Because  of  the  amount of  data  we  have, 4603861 rows,  we're  going  to  list  the  location  classes  and  see  if  it's  possible  to  have  a  model  by  location.


```
cols = [0,1,2]
df1 = df[df.columns[cols]]
classes = df1.drop_duplicates()
```
||location_name| location_latitude|location_longitude |
|-|------------|------------------|-------------------|
|0      |     2021 BLK KINNEY AVE (NW 300ft NW of Lamar)  |30.248691|-97.770409|   
|1      |               CAPITAL OF TEXAS HWY / LAKEWOOD DR|30.371674|-97.785660|  
|2      |400 BLK AZIE MORTON RD (South of Barton Spring...|30.264245|-97.765802|  
|5      |              BURNET RD / PALM WAY (IBM DRIVEWAY)|30.402286|-97.717606|  
|10     | 1000 BLK W CESAR CHAVEZ ST (H&B Trail Underpass)|30.268652|-97.759929| 
|14     |   LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)|30.266800|-97.756051| 
|18     |         100 BLK S CONGRESS AVE (Congress Bridge)|30.259791|-97.746034|  
|22     |           LAMAR BLVD / N LAMAR SB TO W 15TH RAMP|30.279604|-97.750472| 
|31     | CONGRESS AVE / JOHANNA ST (Fulmore Middle Sch...|30.244513|-97.751737| 
|41     |                  1612 BLK S LAMAR BLVD (Collier)|30.250802|-97.765746| 
|45     |                    LAMAR BLVD / SHOAL CREEK BLVD|30.292767|-97.747190| 
|49     |                        700 BLK E CESAR CHAVEZ ST|30.261476|-97.737262| 
|54     |                         LAMAR BLVD / MANCHACA RD|30.243875|-97.781705| 
|60     |          CAPITAL OF TEXAS HWY / WALSH TARLTON LN|30.257986|-97.812248| 
|188    |                           LAMAR BLVD / ZENNIA ST|30.319905|-97.730292| 
|192    |                           BURNET RD / RUTLAND DR|30.383427|-97.724075| 
|269    |                  CAPITAL OF TEXAS HWY / CEDAR ST|30.339468|-97.803558| 
|429    |             3201 BLK S LAMAR BLVD (BROKEN SPOKE)|30.240471|-97.786667| 
|1425963|   LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)|30.266800|-97.756051| 
|1827532|                         LAMAR BLVD / MANCHACA RD|30.243875|-97.781705| 
|2884707|           LAMAR BLVD / N LAMAR SB TO W 15TH RAMP|30.279604|-97.750472| 
|3365977|                    LAMAR BLVD / SHOAL CREEK BLVD|30.292767|-97.747190| 
|3602550|                  CAPITAL OF TEXAS HWY / CEDAR ST|30.339468|-97.803558| 

We can obviously see that we have 23 locations. Then creating a model by every location could be possible and this would be our first try. 

---
## Feature Engineering

Now, we will filter our data to only deal with one location (30.2668: -97.756051) and with one direction (NB).
```
df_Day = df[(df["location_latitude"]==30.2668 ) & (df["location_longitude"]==-97.756051) & (df["Direction"]=='NB')  ].drop(['location_name','location_latitude', 'location_longitude',"Direction"], axis=1)
```

After this step, we have to create a column of type Date that can be set as index. 

```
df_Day = df_Day.sort_values(by=['Year','Month','Day','Hour','Minute'])
df_Day["Date"] = pd.to_datetime((df_Day.Year*10000+df_Day.Month*100+df_Day.Day).apply(str) +" "+df_Day['Time Bin'],format='%Y%m%d %H:%M')
df_Day2= df_Day.set_index(df_Day["Date"]).drop(['Time Bin',"Date"], axis=1)
```

## Exploration

Now, we will plot some useful graphs to better understand the data.

```
df_Day_by_month= df_Day2.resample('M').sum()
sns.lineplot(x=df_Day_by_month.index,y="Volume",data=df_Day_by_month)
plt.savefig('myfigure_110.png', dpi=100)
plt.show()
```

<img  src="screens\1\myfigure_110.png"  width="80%"/>

The first plot illustrates the volume by month, apart from the summer months when the volume is exceptionally low , the volume fluctuates over and under the value of 450 000 with a similar rate except the drop to 250 000 that can be seen at the end of august.

---
```
sns.pointplot(data=df_Day2,x="Hour",y="Volume")
plt.savefig('myfigure_120.png', dpi=100)
plt.show()
```

<img  src="screens\1\myfigure_120.png"  width="80%"/>

By the hour traffic volume fluctuates but increases throughout the day to reach a maximal value of 170 at 7 pm, the volume decreases slowly after that to meet the minimal value in the first hour of the day.

---
```
sns.pointplot(data=df_Day2,x="Day of Week",y="Volume")
plt.savefig('myfigure_130.png', dpi=100)
plt.show()
```

<img  src="screens\1\myfigure_130.png"  width="80%"/>

By day of week traffic increases in business days especially thursday and friday and drops in the weekend

## Preprocessing

### Training and test data :
  After each epoch, the machine will test its learning in relation to the test game. 
This test game is not used in the learning phase, it allows to measure the success rate of the predictions.
 We chose to use  90% of the data for training and 10% for testing.


```
train_size = int(len(df_Day2)*0.9)
test_size = len(df_Day2) - int(len(df_Day2)*0.9)
train,test = df_Day2.iloc[0:train_size], df_Day2.iloc[train_size:len(df_Day2)]
print(train.shape , test.shape )

```


### Robust scaler and transformer:
Robust scaler removes the median and scales the data according to the range of quantiles ((defaults to IQR: Interquartile Range). The IQR is the range between
 the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).  
 Median and interquartile range are kept in order  to be used on later data using the transform method that centers and scales the data.

We’ll scale some of the features we’re using for our modeling:

```
from sklearn.preprocessing import RobustScaler
f_columns = ["Year","Month","Day","Day of Week","Minute"]

f_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:,f_columns] = f_transformer.transform(train[f_columns].to_numpy())
test.loc[:,f_columns] = f_transformer.transform(test[f_columns].to_numpy())

```

We’ll also scale the Volume of cars too:

```ruby
Volume_transformer = RobustScaler()
Volume_transformer = Volume_transformer.fit(train[["Volume"]])
train.loc[:,"Volume"] = Volume_transformer.transform(train["Volume"].to_numpy().reshape(-1, 1))
test.loc[:,"Volume"] = Volume_transformer.transform(test["Volume"].to_numpy().reshape(-1, 1))
```


### Creating the dataset: 
To prepare the sequences, we’re going to reuse the same create_dataset() function:

```
def create_dataset(X,y,time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs),np.array(ys)
```

Each sequence is going to contain 24 data points from the history:

```
TIME_STEPS = 24

X_train, y_train = create_dataset(train,train.Volume,time_steps=TIME_STEPS) 
X_test, y_test = create_dataset(test,test.Volume,time_steps=TIME_STEPS) 
```

## Predicting Counts

### The Model :
   We chose Bidirectional LSTMs  instead of one LSTMs on the input data.The first on the input sequence as-is and the second on a reversed copy of the input sequence. This gives more context to the network and gives faster and even fuller learning to the problem. Bidirectional LSTMs are supported in Keras via the Bidirectional layer wrapper. This wrapper takes a recurrent layer (e.g. the first LSTM layer) as an argument. It also allows you to specify the merge mode, that is how the forward and backward outputs should be combined before being passed on to the next layer. We use here the default option wish is concatenate, and this is the method often used in studies of bidirectional LSTMs.


<img  src="BILSTM.png"  width="100%"/>

```
model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)

model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
```


Here we predict the new counts based on the test sets, you can note that we are using an invers_transform function which converts the scaled data to its original scale.

```
y_pred = model.predict(X_test)

y_train_inv = Volume_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = Volume_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = Volume_transformer.inverse_transform(y_pred)
```

## Evaluation


### Without location (time, volume) :

This is the basic model where neither the location or the direction are given as input for the model.

Here’s what we have after training our model for 30 epochs:

<p float="left">
   <img  src="screens\2\myfigure_100.png"  width="410"/>
   <img  src="screens\2\myfigure_res1.png"  width="410"/>
</p>



That might be too much for your eyes. Let’s zoom in on the predictions:

<p float="left">
   <img  src="screens\2\myfigure_res2.png"  width="410"/>
   <img  src="screens\2\myfigure_res3.png"  width="410"/>
   <img  src="screens\2\myfigure_res4.png"  width="410"/>
</p>

We find that the first model that does not consider location and direction produces a somewhat satisfactory result even though there is an overfitting on epoch 15 where the error on the test set begins to increase while the error on the training set continues to decrease. The best model we can produce with this architecture is the one whose training ends at epoch 15.
### With location (time, latitude, logitude, volume) :

We note that by adding the location to our input data to the model, the model reaches the learning limit at epoch 5. Just like its predecessor, we notice an overfitting in epoch 5 while the error on the training set continues to decrease to 0.07 in epoch 30.

<img  src="screens\3 long and lat included\myfigure_100.png"  width="410"/>

According to the graph below it is easy to notice that the prediction is a little far from the true value of the volume on the test set. Thus, our model needs to be modified to effectively predict this value.

<p float="left">
   <img  src="screens\3 long and lat included\myfigure_res2.png"  width="410"/>
   <img  src="screens\3 long and lat included\myfigure_res3.png"  width="410"/>
   <img  src="screens\3 long and lat included\myfigure_res4.png"  width="410"/>
   <img  src="screens\3 long and lat included\myfigure_res5.png"  width="410"/>
</p>


### All variables included (time, latitude, logitude, direction, volume):

For this example, we will try to use the qualitative variable “Direction”. For this, a preprocessing step is important. We will transform the Direction column to binary columns so that we can use them as quantitative values that can be given as input to the model. Every binary column will mean if a direction is present in a row. Another detail is that we will only use two months of the data because of the huge amount of time and RAM needed to execute the algorithm. 

```
df[['EB', 'NB', 'None', 'SB', 'WB']] = pd.get_dummies(df['Direction'])
df_Day = df[(df["Year"]==2019 ) & (df["Month"]>10)  ].drop(['location_name',"Direction"], axis=1)

```
Out[3]: 

|Date      |location_latitude | location_longitude|Year | ...  |None | SB | WB|
|----------|------------------|-------------------|-----|------|-----|----|---|
|2019-11-01|30.279604         |-97.750472 		  |2019 | ...  |   0 |  1 |  0|
|2019-11-01|30.268652         |-97.759929 		  |2019 | ...  |   0 |  0 |  0|
|2019-11-01|30.268652         |-97.759929 		  |2019 | ...  |   0 |  0 |  0|
|2019-11-01|30.268652         |-97.759929 		  |2019 | ...  |   0 |  0 |  1|
|2019-11-01|30.268652         |-97.759929 		  |2019 | ...  |   0 |  0 |  1|

[5 rows x 14 columns]

<img  src="screens\4 all included\myfigure_100.png"  width="410"/>


The graph above shows that the model is learning quiet well. The test Loss decreases very fast at the beginning and continues to decrease slowly to reach a minimal limit. In addition, the training Loss seems to decrease slowly with no overfitting which means a good model that can predict very well the true value.

<p float="left">
   <img  src="screens\4 all included\myfigure_res2.png"  width="410"/>
   <img  src="screens\4 all included\myfigure_res3.png"  width="410"/>
   <img  src="screens\4 all included\myfigure_res4.png"  width="410"/>
   <img  src="screens\4 all included\myfigure_res5.png"  width="410"/>
</p>

In the above graphs, the prediction seems to identify as good as possible the true value. It is a model that adapts very well to the big variation of the data and makes predictions with the minimal loss.

# A new approach

## Predicting Counts

NeuralProphet is a Neural Network based PyTorch implementation of a user-friendly time series forecasting tool for practitioners. This is heavily inspired by Prophet, which is the popular forecasting tool developed by Facebook. NeuralProphet is developed in a fully modular architecture which makes it scalable to add any additional components in the future.


After making sure our data is in the right format for the neural prophet model. we read the data into a Panda DataFrame. NeuralProphet object expects the time-series data to have a date column named ds and the time-series column value we want to predict as y.

```
df_Day2 = df_Day2[['Date','Volume']].rename(columns={"Date": "ds","Volume":"y"})	
```

Now we can fit an initial model without any customizations. We specify the data frequency to be daily. The model will remember this later when we predict into the future.

```
m = NeuralProphet()	
metrics = m.fit(df_Day2,validate_each_epoch=True, freq="D")	
```

The returned metrics dataframe contains recoded metrics for each training epoch. Next, we create a dataframe to predict on. 

```
future = m.make_future_dataframe(df_Day2, periods=365, n_historic_predictions=len(df_Day2))	
forecast = m.predict(future)
```
The advantage of using this library is its similar syntax to Facebook’s Prophet library.


## Evaluation

You can simply plot the forecast by calling model.plot(forecast) as following:
```
fig1 = m.plot(forecast[1000:1500])	
plt.savefig('fig11.png', dpi=300,bbox_inches="tight")
```
<p float="left">
   <img  src="screens\neural prophet 1 basic\fig11.png"  width="410"/>
   <img  src="screens\neural prophet 1 basic\fig14.png"  width="410"/>
</p>


The one-year forecast plot is shown above, where the time period between 2019-01-25 to 2019-02-02 is the prediction. As can be seen, the forecast plot resembles the historical time-series. It both captured the seasonality as well as the slow-growing linear trend every week.


You can plot the parameters by calling model.plot_parameters().



<img  src="screens\neural prophet 1 basic\fig12.png"  width="100%"/>


We can see that this package gives us all the plots that we did manually with the LSTM model. We can also see that they are pretty much the same results. In addition, we have a trend that would evaluate the evolution of the mean value daily. We see that in the first months of 2019 there is an important trend to capture and use in the model.



The model loss using Mean Absolute Error (MAE) is plotted below. we can also use the Smoothed L1-Loss function.

```
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(metrics["MAE"], 'ob', linewidth=6, label="Training Loss")  
ax.plot(metrics["MAE_val"], '-r', linewidth=2, label="Validation Loss")
plt.legend(loc='best')
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Loss",fontsize=18)
plt.savefig('figres1.png',dpi=300,bbox_inches="tight")
```

<p float="left">
   <img  src="screens\neural prophet 1 basic\figres1.png"  width="410"/>
   <img  src="screens\neural prophet 1 basic\figres2.png"  width="410"/>
</p>

It is noted that the Loss (MAE) of the training step begins very low given the amount of data seen by the model at that time. So it reaches a maximum where it has seen enough data and starts to learn. Then, the Loss decreases to an approximate value of 43 . The Loss on the test data varies somewhat in their amplitude or fluctuation than the training Loss. But, Thetest Loss follows the the training Loss which means a very good learning and a very good model.

The same remarks can be said about the Loss SmouthL1.

the difference between the two approaches is that the first approach (LSTM) seeks to control the variability of the data in order to have the most accurate predictions possible. On the other hand, the second approach (NeuralProphet) seeks to model well a recurring phenomenon and to take in consideration the peak seasons and Seasons where there are fewer cars. According to our analyzes, both approaches have succeeded in accurately predicting the quantity of cars that will pass each 15 minutes knowing the location and direction.

## Conclusion


In conclusion, this study to build a model of prediction of the count of cars will be very important to better understand the flow of cars and anticipate possibilities of stamping and therefore accidents. So this can help to build signaling systems that are more efficient than ever. In addition, these results can be used to anticipate the arrival date and hour of tourists to the city of Austin.


## References
<a id="1">[1]</a> 
[TensorFlow - Time series forecasting](https://www.kaggle.com/vinayshanbhag/radar-traffic-data)
TensorFlow Tutorials

<a id="2">[2]</a> 
[Understanding LSTM Networks](https://www.kaggle.com/vinayshanbhag/radar-traffic-data)
Posted on August 27, 2015

<a id="3">[3]</a> 
[Radar Traffic Data](https://www.kaggle.com/vinayshanbhag/radar-traffic-data)
Traffic data collected from radar sensors deployed by the City of Austin.

<a id="4">[4]</a> 
[Stacked Bidirectional and Unidirectional LSTM
Recurrent Neural Network for
Network-wide Traffic Speed Prediction](https://arxiv.org/ftp/arxiv/papers/1801/1801.02143.pdf)
Zhiyong Cui, Ruimin Ke, Ziyuan Pu, Yinhai Wang


<a id="5">[5]</a> 
[A comparison between LSTM and Facebook Prophet models: a financial forecasting case study](http://openaccess.uoc.edu/webapps/o2/bitstream/10609/107367/7/agonzalez_mataTFG0120memory.pdf)
A comparison between LSTM and Facebook Prophet models: a financial forecasting case study
Alejandro González Mata

<a id="6">[6]</a> 
[Neural Prophet - Extendable and scalable forcasting](https://github.com/ourownstory/neural_prophet/blob/master/notes/Presented_at_International_Symposium_on_Forecasting.pdf)
Oskar Triebe
Stanfrd university october 26,2020
40th International Symposium on Forecasting.

**

# End of Report

**
