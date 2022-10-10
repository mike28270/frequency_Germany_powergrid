import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import seaborn as sns

from keras import backend as K
from stepwindow import WindowGenerator

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot) )

def fillNa(df, show=True):
    if show == True:
        print('Number of missing rows in dataset: \n{}\n'.format(df.isna().sum()))

    # Dealing with null: Fill NA with time interpolate
    if (df.isna().sum()) != 0:
        df = df.interpolate(method ='time')
        if show == True:
            print('Fill null with interpolate...')
            print('Number of missing rows in dataset: \n{}\n'.format(df.isna().sum()))
    # Dealing with null: Fill NA with mean value
        if (df.isna().sum()) != 0:
            df = df.fillna(df.mean())
            if show == True:
                print('Fill null with mean value...')
                print('Number of missing rows in dataset: \n{}\n'.format(df.isna().sum()))
    return df


def normalization(df, show=True, name="Frequency"):
    # Normalization
    if show == True:
        print('Regular Dataset')
        display(df.head())

    scaler = MinMaxScaler()
    a_array = np.array(df).reshape(-1, 1)
    a_array = scaler.fit_transform(a_array)
    df_result = pd.Series(a_array.reshape(-1), index=df.index, name=name)
    #df = (df-df.min())/(df.max()-df.min())

    if show == True:
        print('Normalised Dataset')
        display(df_result.head())

    return df_result, scaler


def preProcessing(df, show=True):
    # Dealing with null
    df = fillNa(df, show)

    # Normalization
    df, scaler = normalization(df, show)

    return df


class trainModel():
    def __init__(self, model, step_window):
        self.model = model
        self.step_window = step_window

    def train(self, epochs=500, patience=2, learning_rate=0.01):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                          patience=patience,
                                                          mode='min')

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           metrics=[tf.keras.metrics.RootMeanSquaredError(), r_squared])

        history = self.model.fit(self.step_window.train,
                            batch_size=1,
                            epochs=epochs,
                            callbacks=[early_stopping],
                            validation_data=self.step_window.val)
        return self.model

    def metric(self):
        val_score = self.model.evaluate(self.step_window.val)
        test_score = self.model.evaluate(self.step_window.test, verbose=0)
        return val_score, test_score

    def plot(self, max_subplots):
        fig, axe = self.step_window.plot(self.model, max_subplots=max_subplots)
        return fig, axe


class performance():
    def __init__(self, score, model):
        self.score = score
        self.model = model

    def metric(self, metric_name):
        val_score = np.array(self.score)[:, 0]
        test_score = np.array(self.score)[:, 1]
        metric_index = self.model.metrics_names.index(metric_name)
        self.val_score = val_score[:, metric_index]
        self.test_score = test_score[:, metric_index]
        return self.val_score, self.test_score

    def plot(self, figsize=(12,5), n_step_interval=1):
        val_mse, test_mse = self.metric(metric_name='root_mean_squared_error')
        val_rsqure, test_rsqure = self.metric(metric_name='r_squared')

        plot_dict = {'rmse': [val_mse, test_mse],
                     'r-square': [val_rsqure, test_rsqure]}

        fig, axe = plt.subplots(1, 2, figsize=(12,5))
        for i, [key, dataset] in enumerate(plot_dict.items()):
            axe[i].plot(dataset[0], label='Validation')
            axe[i].plot(dataset[1], label='Test')
            axe[i].legend(ncol=2, loc="upper right", fontsize=16)
            axe[i].set_title(key)
            axe[i].set_xticks(range(0, len(val_mse), n_step_interval))
            axe[i].set_xticklabels(range(1, len(val_mse)+1, n_step_interval))
            axe[i].set_xticks(range(0, len(val_mse), 1), minor=True)
            #axe[i].set_xticklabels(range(1, len(val_mse)+1, 1), minor=True)
            axe[i].set_xlabel("n_step", size=15)
            axe[i].tick_params(axis='x', width=1, length=7,
                            direction='inout', rotation=0, labelsize=13, right=True)
            axe[i].tick_params(axis='y', width=1, length=7,
                            direction='inout', rotation=0, labelsize=13, labelleft=True)
            axe[i].grid(which='major', alpha=0.8)
            axe[i].grid(which='minor', alpha=0.4)
        return fig, axe


class predictModel():
    def __init__(self, df_original):
        self.df_original = pd.DataFrame(df_original)

    # (column_name: low_limit=-55, high_limit=55)
    def preprocessing(self, limits):
        df_actual = self.df_original.copy()
        for column_name, low_limit, high_limit in limits:
            # Deal with outliner
            outliner_rows = (df_actual[column_name]<low_limit) | (df_actual[column_name]>high_limit)
            if (outliner_rows.any()) == True:
                #print("Outliners deleted")
                index = (df_actual.loc[outliner_rows]).index
                df_actual.loc[index] = None

            # Pre-processing
            df_actual[column_name] = fillNa(df_actual[column_name], False)
            df_actual[column_name], self.scaler = normalization(df_actual[column_name], False)

        self.df_actual = df_actual
        return self.df_actual, self.scaler

    def split_sequence(self, input_width, shift):
        X, y = list(), list()
        self.shift = shift
        for i in range(len(self.df_actual)):
            # find the end of this pattern
            end_ix = i + input_width
            # check if we are beyond the sequence
            if end_ix > len(self.df_actual)-shift:
                break
            # gather input and output parts of the pattern
            seq_x = self.df_actual.iloc[i:end_ix],
            seq_y = self.df_actual.iloc[end_ix:(end_ix+shift)]
            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)
        self.df_actual_X = X.reshape((X.shape[0], input_width, 1))
        self.df_actual_y = y.reshape((y.shape[0], shift, 1))
        self.df_original_y = self.df_original[input_width:].copy()
        return self.df_actual_X, self.df_actual_y

    def predict(self, model_list):
        self.df_pred_y = model_list[self.shift-1].predict(self.df_actual_X)
        self.df_pred_y = np.reshape(self.df_pred_y, (self.df_actual_X.shape[0], self.shift))
        self.df_actual_y = np.reshape(self.df_actual_y, (self.df_actual_y.shape[0], self.shift))
        return self.df_pred_y, self.df_actual_y

    def metric(self):
        self.rmse = math.sqrt(metrics.mean_squared_error(self.df_actual_y, self.df_pred_y))
        self.r2score = r2_score(self.df_actual_y, self.df_pred_y)
        return self.rmse, self.r2score

    def metric_plot(self, score, axe, n_step_interval=1, legends=None):
        #fig, axe = plt.subplots(1, 2, figsize=(20,5))
        title = ['RMSE', 'R-Squared']
        score_arr = np.array(score)
        for i in range(score_arr.shape[1]):
            score_plot = score_arr[:, i]
            axe[i].plot(score_plot)
            axe[i].set_xticks(range(0, len(score_plot), 1))
            axe[i].set_xticklabels(range(1, len(score_plot)+1, 1))
            axe[i].set_xlabel("n_step", size=13)
            if legends != None:
                axe[i].legend(legends, title='Input Size', title_fontsize=13, fontsize=13)

            axe[i].set_xticks(range(0, len(score), n_step_interval))
            axe[i].set_xticklabels(range(1, len(score)+1, n_step_interval))
            axe[i].set_xticks(range(0, len(score), 1), minor=True)
            #axe[i].set_xticklabels(range(1, len(val_mse)+1, 1), minor=True)
            axe[i].tick_params(axis='x', width=1, length=7,
                            direction='inout', rotation=0, labelsize=13, right=True)
            axe[i].tick_params(axis='y', width=1, length=7,
                            direction='inout', rotation=0, labelsize=13, labelleft=True)
            axe[i].grid(which='major', alpha=0.8)
            axe[i].grid(which='minor', alpha=0.4)
            axe[i].grid(visible=True, which='major', axis='y')
        return axe

    def plot(self, axe, ylabel="Frequency [mHz]", title=" ", legendsize=10, showmetric=True):
        def annotate(**kws):
            bbox = dict(boxstyle="round,pad=0.3", alpha=0.7, fc="white", ec="grey", lw=1)
            annotate_value = (f"RMSE: {self.rmse:0.4f}\n"
                              f"R-Squared: {self.r2score:0.4f}")
            axe.annotate(annotate_value, xy=(0.73, 0.1),
                         xycoords=axe.transAxes, color='black', bbox=bbox, fontsize=13)
        # Calculate metric
        self.metric()

        # Original Data
        df_original_y = pd.DataFrame(self.df_original_y)

        # Put Datetime index to plot
        df_pred_y_list = []
        for i in range(0, self.df_pred_y.shape[1]):
            index = self.df_original_y[i:(df_original_y.shape[0]-self.shift+1+i)].index
            df_pred_y_temp = self.scaler.inverse_transform(self.df_pred_y)
            df_pred_y_temp = pd.DataFrame(df_pred_y_temp[:, i], index=index)
            df_pred_y_list.append(df_pred_y_temp)
        df_pred_y = pd.DataFrame(pd.concat(df_pred_y_list, axis=1).mean(axis=1))

        # Plot
        axe.plot(df_original_y, label="Actual")
        axe.plot(df_pred_y, label="Prediction")
        axe.set_title(title, size=14)
        axe.set_ylabel(ylabel,  size=13)
        axe.legend(ncol=2, loc="upper right", fontsize=legendsize)

        axe.tick_params(axis='x', width=1, length=7,
                        direction='inout', rotation=0, labelsize=13, right=True)
        axe.tick_params(axis='y', width=1, length=7,
                        direction='inout', rotation=0, labelsize=13, labelleft=True)

        axe.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        axe.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y\n    %m"))
        axe.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
        #axe.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%d"))
        axe.grid(visible=True, which='major', axis='x', alpha=0.8)
        axe.grid(visible=True, which='minor', axis='x', alpha=0.5)
        axe.grid(visible=True, which='major', axis='y', alpha=0.8)
        if showmetric == True:
            annotate()
        return axe
