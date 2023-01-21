import shutil
import stat
import errno
from rest_framework.response import Response
# Django Libraries
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, Http404
from django.shortcuts import redirect
from django.conf import settings
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.core.files.storage import FileSystemStorage
from django.templatetags.static import static

from pandas import DataFrame
import sys
import os
import csv
import pandas as pd
import numpy as np
import category_encoders as ce

# Sklearn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs


# Matplot
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing._encoders import OrdinalEncoder
import time


def iv_woe(data, target, bins, show_woe):

    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'],
                                          0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        # print("Information value of " + ivars +
        #       " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [
                            d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        # print("******************************New Dataframe****************************")
        # print(newDF)
        # print("******************************Woe Dataframe****************************")
        # print(woeDF)


def get_NaN_percent(fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = (df.isnull().sum() * 100 / len(df)
                   ).sum() / len(clm_list)
    NaN_percent = NaN_percent.round(2)
    return NaN_percent


def Overview(fName):

    start = time.time()
    df = get_df(fName)
    file_path = os.path.join(settings.MEDIA_ROOT, 'processed/'+fName+'.csv')
    statInfo = os.stat(file_path)
    fileSize = statInfo.st_size
    fileSize = fileSize // 1000
    clm_list = list(df)

    # print("----------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------")

    # df2 = get_df(fName)
    # df2 = df2['BP'].astype(pd.Int64Dtype())  # dataset with missing values
    # df = df[df['BP'].notna()]  # dataset without missing values
    # df['BP'] = df['BP'].astype(pd.Int64Dtype())

    # X, y = make_blobs(n_samples=df['BP'].head(
    #     100),  n_features=2, random_state=1)

    # # # fit final model
    # model = LogisticRegression()
    # model.fit(X, y)
    # # # new instances where we do not know the value
    # Xnew, _ = make_blobs(n_samples=(len(df2))-len(df['BP']))
    # # # make a prediction
    # ynew = model.predict(Xnew)
    # # show the inputs and predicted outputs
    # for i in range(len(df2)-len(df['BP'])):
    #     print("Predicted ", ynew[i])

    # print("----------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------")

    # datatype
    # ========

    dataType_list = df.dtypes

    categorical_clms_lst = []
    date_time_clms_lst = []
    numerical_clms_lst = []

    cols = list(df)

    for i in clm_list:
        if 'date' in i.lower() or '_dt' in i.lower():
            date_time_clms_lst.append(i)
        elif df[i].dtypes == 'int64' or df[i].dtypes == 'float64':
            numerical_clms_lst.append(i)
        else:
            categorical_clms_lst.append(i)

    for date_time_col in date_time_clms_lst:
        df[date_time_col] = pd.to_datetime(df[date_time_col], dayfirst=True)

    categorical_clms = len(categorical_clms_lst)
    date_time_clms = len(date_time_clms_lst)
    numerical_clms = len(numerical_clms_lst)

    if categorical_clms <= 0:
        categorical_msg = "Categorical Features Does Not Exits"
    else:
        categorical_msg = ""

    if numerical_clms <= 0:
        numerical_msg = "Numerical Features Does Not Exits"
    else:
        numerical_msg = ""

    if date_time_clms <= 0:
        date_time_msg = "Date-Time Features Does Not Exits"
    else:
        date_time_msg = ""

    # No of rows and columns
    # ======================
    data_frame = pd.read_csv(os.path.join(settings.MEDIA_ROOT,
                                          'processed/'+fName+'.csv'))
    rows = len(data_frame.index)
    columns = len(list(df))

    # NaN Values
    # ==========
    NaN_percent = get_NaN_percent(fName)
    total_Nan = (df.isnull().sum(axis=0)).sum()

    zippend_list = zip(clm_list, dataType_list)

    context = {
        'fName': fName,
        'fSize': fileSize,
        'rows': rows,
        'clm_list': clm_list,
        'columns': columns,
        'zip': zippend_list,
        'total_NaN': total_Nan,
        'NaN_percent': NaN_percent,
        'categorical': categorical_clms,
        'numerical': numerical_clms,
        'datetime': date_time_clms,
        'cat_list': categorical_clms_lst,
        'num_list': numerical_clms_lst,
        'date_time_list': date_time_clms_lst,
        'cat_msg': categorical_msg,
        'num_msg': numerical_msg,
        'date_time_msg': date_time_msg,
    }

    end = time.time()
    print("Execution time of Overview", end - start)

    context['undo_count'] = changesCount(fName)
    context['execution_time'] = end - start

    return context


def Upload(request):
    start = time.time()
    if request.method == 'POST':
        uploaded_file = request.FILES['dataset']
        arr = uploaded_file.name.split('.', 1)
        fName = arr[0]+'_'+str(0)
        extension = arr[1]
        fullName = fName+'.'+extension

        # Validating the uploaded file
        if extension == 'csv' or extension == 'xls' or extension == 'xlsx':

            # if os.path.exists(os.path.join(settings.MEDIA_ROOT, 'original/') and os.path.join(settings.MEDIA_ROOT, 'processed/')):
            #     shutil.rmtree(os.path.join(
            #         settings.MEDIA_ROOT, 'original/'), ignore_errors=False,
            #         onerror=handleRemoveReadonly)
            #     shutil.rmtree(os.path.join(
            #         settings.MEDIA_ROOT, 'processed/'), ignore_errors=False,
            #         onerror=handleRemoveReadonly)
            try:
                os.mkdir(os.path.join(settings.MEDIA_ROOT, '/original'))
                os.mkdir(os.path.join(settings.MEDIA_ROOT, '/processed'))
            except:
                pass

            fs1 = FileSystemStorage()
            fs1.save('original/'+fullName, uploaded_file)
            fs2 = FileSystemStorage()
            fs2.save('processed/'+fullName, uploaded_file)

            file_path1 = os.path.join(
                settings.MEDIA_ROOT, 'original/'+fullName)
            file_path2 = os.path.join(
                settings.MEDIA_ROOT, 'processed/'+fullName)

            # converting xls, xlsx to csv
            if extension == 'xls' or extension == 'xlsx':
                df = pd.read_excel(os.path.join(settings.MEDIA_ROOT,
                                                'processed/'+fullName), engine="openpyxl")
                df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                       'processed/'+fName+'.csv'), index=False)

            df = pd.read_csv(os.path.join(settings.MEDIA_ROOT,
                                          'processed/'+fName+'.csv'))
            df = df.replace(to_replace="?",
                            value="nan")
            df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                   'processed/'+fName+'.csv'), index=False)

            context = Overview(fName)
            context['status'] = 'Success'
            context['message'] = 'Dataset Uploaded Successfully'
            return render(request, 'index.html', context)
        else:
            context = {
                'fName': fName,
                'status': 'Error',
                'message': 'Please upload .csv, .xls files'
            }
            return render(request, 'Dataset/Upload.html', context)

    end = time.time()
    print("Execution time for Uploading Dataset", end - start)

    return render(request, 'Dataset/Upload.html')


# routes
# ======

def Home(request, fName):
    context = Overview(fName)
    return render(request, 'index.html',  context)


def RemoveUnwantedFeatures(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN(fName)

    drop_nan = zip(clm_list, NaN_percent)
    drop_col = zip(clm_list, NaN_percent)

    nan_percent = get_NaN_percent(fName)

    context = {
        'fName': fName,
        'attr_drop_list': drop_nan,
        'attr_drop_col_list': drop_col,
        'NaN_percent': nan_percent,
    }
    context['undo_count'] = changesCount(fName)

    return render(request, 'Imputations/DropUnwantedFeature.html', context)


def RemoveUnwantedFeaturesCalc(request, fName):
    df = get_df(fName)

    if request.method == 'POST':
        start = time.time()
        selected_col = request.POST.getlist('dropFeatures')
        df.drop(selected_col, axis=1, inplace=True)

        fName = currentFname(fName)

        df.to_csv(os.path.join(settings.MEDIA_ROOT,
                               'processed/'+fName+'.csv'), index=False)

        clm_list = list(df)
        NaN_percent = get_NaN(fName)
        drop_nan = zip(clm_list, NaN_percent)
        drop_col = zip(clm_list, NaN_percent)

        nan_percent = get_NaN_percent(fName)
        end = time.time()
        print("Execution time for Removing unwanted features", end - start)
        context = {
            'fName': fName,
            'attr_drop_list': drop_nan,
            'attr_drop_col_list': drop_col,
            'NaN_percent': nan_percent,
            'status': 'Success',
            'message': "Selected features are dropped. Please refresh the page and see the changes.",
            'execution_time': end - start,
        }
        return render(request, 'Imputations/DropUnwantedFeature.html', context)

    return HttpResponse("Error ! Please go back.")


def getVisualization(fName):
    df = get_df(fName)
    clm_list = []
    corr_list = []
    for corr in df.corr(numeric_only=False).values:
        corr_list.append(list(corr))
    cat_clm_list = []
    num_clm_list = []
    dt_clm_list = []
    for i in list(df):
        if df[i].dtype == 'int64' or df[i].dtype == 'float64':
            num_clm_list.append(i)
        else:
            cat_clm_list.append(i)
    nan_percent = get_NaN_percent(fName)

    if len(cat_clm_list) <= 0:
        categorical_msg = "Categorical Features Does Not Exits"
    else:
        categorical_msg = ""

    if len(num_clm_list) <= 0:
        numerical_msg = "Numerical Features Does Not Exits"
    else:
        numerical_msg = ""

    if len(dt_clm_list) <= 0:
        date_time_msg = "Date-Time Features Does Not Exits"
    else:
        date_time_msg = ""

    context = {
        'fName': fName,
        'clm_list': num_clm_list,
        'NaN_percent': nan_percent,
        "custom_chart_status": '',
    }

    return context


def Dataset(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    values = df.values

    paginator = Paginator(values, 200)
    page = request.GET.get('page', 1)
    try:
        data = paginator.page(page)
    except PageNotAnInteger:
        data = paginator.page(1)
    except EmptyPage:
        data = paginator.page(paginator.num_pages)

    context = {
        'fName': fName,
        'clm_list': clm_list,
        'for_filter': list(df),
        'integrated_ds': False,
        'values': data,
    }
    context['undo_count'] = changesCount(fName)

    return render(request, 'Dataset/Dataset.html', context)


def OriginalDataset(request, fName):
    res = fName.rsplit('_', 1)
    step = int(res[1])
    if step == 0:
        pass
    else:
        fName = res[0]+'_0'
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT,
                                  'original/'+fName+'.csv'))
    clm_list = list(df)
    values = df.values

    paginator = Paginator(values, 200)
    page = request.GET.get('page', 1)
    try:
        data = paginator.page(page)
    except PageNotAnInteger:
        data = paginator.page(1)
    except EmptyPage:
        data = paginator.page(paginator.num_pages)

    context = {
        'fName': fName,
        'clm_list': clm_list,
        'for_filter': list(df),
        'values': data,
    }
    context['undo_count'] = changesCount(fName)

    return render(request, 'Dataset/OriginalDataset.html', context)


def Explore(request, fName):
    start = time.time()
    df = get_df(fName)

    # NaN percent
    nan_percent = get_NaN_percent(fName)

    clm_list = list(df)

    # explore

    mean_list = get_mean(fName)
    median_list = get_median(fName)
    std_list = get_std(fName)

    kurt_list = kurtosis(fName)
    skew_list = skewness(fName)

    # NaN_Percentage
    NaN_values = df.isnull().sum(axis=0)
    NaN_list = get_NaN(fName)
    NaN_list = NaN_list.round(2)
    NaN_list_zip = zip(clm_list, NaN_values, NaN_list)

    new_mean_list = []
    new_median_list = []
    new_std_list = []
    new_skew_list = []
    new_kurt_list = []
    new_nan__list = (df.isnull().sum()).round(2)

    for i in clm_list:
        if df[i].dtype == "float64" or df[i].dtype == "int64":
            new_mean_list.append(df[i].mean())
            new_median_list.append(df[i].median())
            new_std_list.append(df[i].std())
            new_skew_list.append(df[i].skew())
            new_kurt_list.append((df[i].kurt(axis=None, skipna=True)))

        else:
            new_mean_list.append('-')
            new_median_list.append('-')
            new_std_list.append('-')
            new_skew_list.append('-')
            new_kurt_list.append('-')

    pack = zip(
        clm_list,
        new_mean_list,
        new_median_list,
        new_std_list,
        new_skew_list,
        new_kurt_list,
        new_nan__list)

    end = time.time()
    print("Execution time for Exploring Dataset", end - start)

    context = {
        'fName': fName,
        'pack': pack,
        'kurtosis_list': kurt_list,
        'skewness_list': skew_list,
        'clm_list': clm_list,
        'NaN_list': NaN_list_zip,
        'NaN_percent': nan_percent,
        'mean_list': mean_list,
        'median_list': median_list,
        'std_list': std_list,
        'execution_time': end - start
    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'Exploration.html', context)


def AttrDropNan(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN(fName)

    drop_nan = zip(clm_list, NaN_percent)
    drop_col = zip(clm_list, NaN_percent)

    nan_percent = get_NaN_percent(fName)

    context = {
        'fName': fName,
        'attr_drop_list': drop_nan,
        'attr_drop_col_list': drop_col,
        'NaN_percent': nan_percent,
    }
    context['undo_count'] = changesCount(fName)

    return render(request, 'Imputations/AttrDropNan.html', context)


def AttrDropNanCalc(request, fName):

    df = get_df(fName)

    clm_list = list(df)
    NaN_percent = get_NaN(fName)
    drop_nan = zip(clm_list, NaN_percent)
    drop_col = zip(clm_list, NaN_percent)

    nan_percent = get_NaN_percent(fName)

    if request.method == 'POST':
        start = time.time()
        selected_col = request.POST.getlist('attrDropCols')
        for single_col in selected_col:
            df = df.dropna(subset=[single_col])

        fName = currentFname(fName)

        df.to_csv(os.path.join(settings.MEDIA_ROOT,
                               'processed/'+fName+'.csv'), index=False)
        end = time.time()
        print("Execution time of AttrDropNan()", end - start)
        context = {
            'fName': fName,
            'attr_drop_list': drop_nan,
            'attr_drop_col_list': drop_col,
            'NaN_percent': nan_percent,
            'status': 'Success',
            'message': "NaN values are dropped. Please refresh the page and see the changes.",
            'execution_time': end - start
        }

        context['undo_count'] = changesCount(fName)

        return render(request, 'Imputations/AttrDropNan.html', context)

    return HttpResponse("Error ! Please go back.")


def AttrDropColCalc(request, fName):
    df = get_df(fName)

    if request.method == 'POST':
        start = time.time()
        selected_col = request.POST.getlist('attrDropCompleteCols')
        df.drop(selected_col, axis=1, inplace=True)

        fName = currentFname(fName)

        df.to_csv(os.path.join(settings.MEDIA_ROOT,
                               'processed/'+fName+'.csv'), index=False)

        clm_list = list(df)
        NaN_percent = get_NaN(fName)
        drop_nan = zip(clm_list, NaN_percent)
        drop_col = zip(clm_list, NaN_percent)

        nan_percent = get_NaN_percent(fName)

        end = time.time()
        print("Execution time of AttrDropCol()", end - start)

        context = {
            'fName': fName,
            'attr_drop_list': drop_nan,
            'attr_drop_col_list': drop_col,
            'NaN_percent': nan_percent,
            'status': 'Success',
            'message': "Selected columns are dropped. Please refresh the page and see the changes.",
            'execution_time': end - start
        }

        context['undo_count'] = changesCount(fName)

        return render(request, 'Imputations/AttrDropNan.html', context)

    return HttpResponse("Error ! Please go back.")


def CompleteDropNan(request, fName):
    start = time.time()
    df = get_df(fName)
    clm_list = list(df)
    for col in clm_list:
        df[col] = df[col].replace('-', np.nan)
        df = df.dropna(axis=0, subset=[col])

    fName = currentFname(fName)

    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                           'processed/'+fName+'.csv'), index=False)

    end = time.time()
    print("Execution time of CompleteDropNan()", end - start)
    context = Overview(fName)
    context['status'] = 'Success'
    context['message'] = 'All the NaN values are dropped'
    context['execution_time'] = end - start

    return render(request, 'index.html', context)


def AttrFillNan(request, fName):
    df = get_df(fName)
    NaN_percent = get_NaN(fName)
    clm_list = list(df)
    attr_fill = zip(clm_list, NaN_percent)

    nan_percent = get_NaN_percent(fName)

    context = {
        'fName': fName,
        'NaN_percent': nan_percent,
        'attr_fill_list': attr_fill,
    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'Imputations/AttrFillNan.html', context)


def AttrFillNanCalc(request, fName):

    if request.method == "POST":
        start = time.time()
        df = get_df(fName)
        status = ''

        selectOption = request.POST.get('fillnaMethods')

        selectedCols = request.POST.getlist('attrFillCols')

        fName = currentFname(fName)

        if selectedCols:
            if selectOption == "fill":
                fillType = request.POST.get('fillType')
                # forward fill
                if fillType == 'ffill':
                    for col in selectedCols:
                        df[col].fillna(method=fillType, inplace=True)

                    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                           'processed/'+fName+'.csv'), index=False)
                    status = 'Success'
                    message = 'NaN values of selected columns are filled by Forward method. Please refresh the page and see the changes.'
                # backward fill
                elif fillType == 'bfill':
                    for col in selectedCols:
                        df[col].fillna(method=fillType, inplace=True)
                    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                           'processed/'+fName+'.csv'), index=False)
                    status = 'Success'
                    message = 'NaN values of selected columns are filled by Backward method.'
                elif fillType == 'mode':
                    for col in selectedCols:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                           'processed/'+fName+'.csv'), index=False)
                    status = 'Success'
                    message = 'NaN values of selected columns are filled by Mode method.'
                elif fillType == 'mean':
                    for col in selectedCols:
                        df[col].fillna((df[col].mean()).round(2), inplace=True)
                    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                           'processed/'+fName+'.csv'), index=False)
                    status = 'Success'
                    message = 'NaN values of selected columns are filled by Mean values.'

                else:
                    pass

            elif selectOption == "replace":
                replaceWord = request.POST.get('replaceBy')
                for col in selectedCols:
                    df[col].fillna(replaceWord, inplace=True)
                df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                       'processed/'+fName+'.csv'), index=False)
                status = 'Success'
                message = 'NaN values of selected columns are replaced by '+replaceWord

            elif selectOption == "interpolate":
                pass

        else:
            status = 'Alert'
            message = 'Please Choose atleast one feature for Fill NaN.'

        end = time.time()
        print("Execution time of AttrFillNan()", end - start)

        NaN_percent = get_NaN(fName)
        nan_percent = get_NaN_percent(fName)
        clm_list = list(df)
        attr_fill = zip(clm_list, NaN_percent)
        nan_percent = get_NaN_percent(fName)
        context = {
            'fName': fName,
            'NaN_percent': nan_percent,
            'attr_fill_list': attr_fill,
            'status': status,
            'message': message,
            'execution_time': end - start,
        }
        context['undo_count'] = changesCount(fName)

        return render(request, 'Imputations/AttrFillNan.html', context)

    return HttpResponse("Error ! Go back.")


# undo function start
def currentFname(filename):
    result_array = filename.rsplit('_', 1)
    step = int(result_array[1]) + 1
    currentFname = result_array[0]+'_'+str(step)
    return currentFname


def changesCount(filename):
    result_array = filename.rsplit('_', 1)
    changesCount = int(result_array[1])
    return changesCount


def Undo(request, fName):
    result_array = fName.rsplit('_', 1)
    current_step = int(result_array[1])
    if current_step > 0:
        step = current_step - 1
        fName = result_array[0]+'_'+str(step)
        context = Overview(fName)
        context['status'] = 'Success'
        context['message'] = 'Your recent action is rolled back successfully.'

        return render(request, 'index.html', context)

    else:
        print('Currently no changes in your dataset')

# undo function end


def Binning(request, fName):

    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    bin_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            bin_list.append(clm)
        else:
            pass
    binning_list = []
    binned_list = []
    for col_name in bin_list:
        if 'bins' in col_name:
            binned_list.append(col_name)
        else:
            binning_list.append(col_name)
    context = {
        'fName': fName,
        'binning_list': binning_list,
        'binned_list': binned_list,
        'NaN_percent': NaN_percent,

    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/Binning.html', context)


def BinningCalc(request, fName):
    df = get_df(fName)

    if request.method == "POST":

        start = time.time()
        selectedCols = request.POST.getlist('binCol')
        binRange = request.POST.get('rangeVal')
        binType = request.POST.get('binningType')

        # check bin range
        if binRange != '':
            pass
        else:
            binRange = 10

        for col in selectedCols:
            dt = df[col].dtype
            if dt == 'float64':
                df[col] = df[col].round()
                df[col] = df[col].astype(int)
                df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                       'processed/'+fName+'.csv'), index=False)
            else:
                pass

        for selected_col in selectedCols:
            bins = []
            labels = []
            Min = int(min(df[selected_col]))
            Max = int(max(df[selected_col]))

            # binning starts

            for i in range(Min, Max, int(binRange)):
                bins.append(i)
            if Max not in bins:
                bins.append(Max)
            l1 = len(bins)
            for j in range(1, l1):
                labels.append(j)

            if binType == 'qcut':
                df[selected_col] = pd.qcut(df[selected_col], q=binRange,
                                           duplicates='drop')
            else:
                df[selected_col] = pd.cut(df[selected_col], bins=bins,
                                          labels=labels, include_lowest=True)
                df[selected_col].fillna(method='bfill', inplace=True)
            # binning ends
        fName = currentFname(fName)

        df.to_csv(os.path.join(settings.MEDIA_ROOT,
                               'processed/'+fName+'.csv'), index=False)

        end = time.time()
        print("Execution time of Binning", end - start)

        df_new = get_df(fName)
        clm_list = list(df_new)
        NaN_percent = get_NaN_percent(fName)
        bin_list = []
        for clm in clm_list:
            dt = df_new[clm].dtype
            if dt == 'int64' or dt == 'float64':
                bin_list.append(clm)
            else:
                pass

        context = {
            'fName': fName,
            'binning_list': bin_list,
            'binned_list': selectedCols,
            'NaN_percent': NaN_percent,
            'status': 'Success',
            'message': 'Binning was done on selected features. Please go to the dataset and see the changes.',
            'execution_time': end - start
        }

        context['undo_count'] = changesCount(fName)

        return render(request, 'FeatureEngineering/Binning.html', context)

    return HttpResponse("Error ! Please go back.")


def LabelEncoding(request, fName):

    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    labelling_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            pass
        else:
            labelling_list.append(clm)

    labelled_list = []
    for col_name in clm_list:
        if 'label' in col_name:
            labelled_list.append(col_name)
        else:
            pass
    context = {
        'fName': fName,
        'labelling_list': labelling_list,
        'labelled_list': labelled_list,
        'NaN_percent': NaN_percent,

    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/LabelEncoding.html', context)


def LabelEncodingCalc(request, fName):

    df = get_df(fName)

    label_encoder = LabelEncoder()
    fName = currentFname(fName)

    if request.method == 'POST':
        start = time.time()
        selected_cols = request.POST.getlist('labelCol')
        for selected_col in selected_cols:
            df[selected_col] = label_encoder.fit_transform(
                df[selected_col].astype(str))
        df.to_csv(os.path.join(settings.MEDIA_ROOT,
                               'processed/'+fName+'.csv'), index=False)

        df_new = get_df(fName)
        clm_list = list(df_new)
        NaN_percent = get_NaN_percent(fName)
        label_list = []
        for clm in clm_list:
            dt = df_new[clm].dtype
            if dt == 'int64' or dt == 'float64':
                pass
            else:
                label_list.append(clm)

        end = time.time()
        print("Execution time of LabelEncoding", end - start)
        context = {
            'fName': fName,
            'labelling_list': label_list,
            'labelled_list': selected_cols,
            'NaN_percent': NaN_percent,
            'status': 'Success',
            'message': 'Label Encoding was done on selected features.',
            'execution_time': end - start
        }
        context['undo_count'] = changesCount(fName)

        return render(request, 'FeatureEngineering/LabelEncoding.html', context)

    return HttpResponse("Error ! Please go back.")


def OneHotEncoding(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    oneHot_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            pass
        else:
            oneHot_list.append(clm)

    oneHotProcessed_list = []
    for col_name in clm_list:
        if 'onehot' in col_name:
            oneHotProcessed_list.append(col_name)
        else:
            pass

    context = {
        'fName': fName,
        'processing_list': oneHot_list,
        'processed_list': oneHotProcessed_list,
        'NaN_percent': NaN_percent,

    }
    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/OneHotEncoding.html', context)


def OneHotEncodingCalc(request, fName):
    df = get_df(fName)
    fName = currentFname(fName)

    if request.method == 'POST':
        start = time.time()
        selected_cols = request.POST.getlist('oneHotCol')
        drop_column = request.POST.get('drop-column')
        for selected_col in selected_cols:
            dummies = pd.get_dummies(df[selected_col], prefix=selected_col)
            df = pd.concat([df, dummies], axis='columns')
            ans = df[selected_col].value_counts(normalize=True) * 100
            # drop column
            del df[selected_col]
            df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                   'processed/'+fName+'.csv'), index=False)

        df_new = get_df(fName)
        clm_list = list(df_new)
        NaN_percent = get_NaN_percent(fName)
        oneHot_list = []
        for clm in clm_list:
            dt = df_new[clm].dtype
            if dt == 'int64' or dt == 'float64':
                pass
            else:
                oneHot_list.append(clm)

        end = time.time()
        print("Execution time of OneHotEncoding", end - start)
        context = {
            'fName': fName,
            'processing_list': oneHot_list,
            'processed_list': selected_cols,
            'NaN_percent': NaN_percent,
            'status': 'Success',
            'message': 'One-Hot Encoding was done on selected features.',
            'execution_time': end - start

        }

        context['undo_count'] = changesCount(fName)

        return render(request, 'FeatureEngineering/OneHotEncoding.html', context)


def OrdinalEncoding(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    ordinal_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            pass
        else:
            ordinal_list.append(clm)

    ordinalProcessed_list = []
    for col_name in clm_list:
        if 'ordinal' in col_name:
            ordinalProcessed_list.append(col_name)
        else:
            pass

    context = {
        'fName': fName,
        'processing_list': ordinal_list,
        'processed_list': ordinalProcessed_list,
        'NaN_percent': NaN_percent,

    }
    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/OrdinalEncoding.html', context)


def OrdinalEncodingCalc(request, fName):
    df = get_df(fName)
    fName = currentFname(fName)

    if request.method == 'POST':
        selected_cols = request.POST.getlist('ordinalCol')

        start = time.time()

        # ordinal calc
        enc = OrdinalEncoder()
        df[selected_cols] = enc.fit_transform(df[selected_cols])
        df.to_csv(os.path.join(settings.MEDIA_ROOT,
                               'processed/'+fName+'.csv'), index=False)

        df_new = get_df(fName)
        clm_list = list(df_new)
        NaN_percent = get_NaN_percent(fName)
        ordinal_list = []
        for clm in clm_list:
            dt = df_new[clm].dtype
            if dt == 'int64' or dt == 'float64':
                pass
            else:
                ordinal_list.append(clm)

        end = time.time()
        print("Execution time of OrdinalEncoding", end - start)

        context = {
            'fName': fName,
            'processing_list': ordinal_list,
            'processed_list': selected_cols,
            'NaN_percent': NaN_percent,
            'status': 'Success',
            'message': 'Ordinal Encoding was done on selected features.',
            'execution_time': end - start

        }

        context['undo_count'] = changesCount(fName)

        return render(request, 'FeatureEngineering/OrdinalEncoding.html', context)


def BinaryEncoding(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    binary_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            pass
        else:
            binary_list.append(clm)

    binaryProcessed_list = []

    context = {
        'fName': fName,
        'processing_list': binary_list,
        'processed_list': binaryProcessed_list,
        'NaN_percent': NaN_percent,

    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/BinaryEncoding.html', context)


def BinaryEncodingCalc(request, fName):
    df = get_df(fName)
    fName = currentFname(fName)

    if request.method == 'POST':
        start = time.time()
        selected_cols = request.POST.getlist('binaryCol')
        for selected_col in selected_cols:
            encoder = ce.BinaryEncoder(cols=[selected_col])
            df = encoder.fit_transform(df)
            df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                   'processed/'+fName+'.csv'), index=False)
        df_new = get_df(fName)
        clm_list = list(df_new)
        NaN_percent = get_NaN_percent(fName)
        binary_list = []
        for clm in clm_list:
            dt = df_new[clm].dtype
            if dt == 'int64' or dt == 'float64':
                pass
            else:
                binary_list.append(clm)

        end = time.time()
        print("Execution time of BinaryEncoding", end - start)

        context = {
            'fName': fName,
            'processing_list': binary_list,
            'processed_list': selected_cols,
            'NaN_percent': NaN_percent,
            'status': 'Success',
            'message': 'Binary Encoding was done on selected features.',
            'execution_time': end - start

        }

        context['undo_count'] = changesCount(fName)

        return render(request, 'FeatureEngineering/BinaryEncoding.html', context)


def CountFrequencyEncoding(request, fName):
    df = get_df(fName)
    NaN_percent = get_NaN_percent(fName)
    clm_list = list(df)

    CF_Processed_list = []
    for col_name in clm_list:
        if 'cf' in col_name:
            CF_Processed_list.append(col_name)
        else:
            pass
    CF_list = list(set(clm_list) - set(CF_Processed_list))
    context = {
        'fName': fName,
        'cf_processing_list': CF_list,
        'cf_processed_list': CF_Processed_list,
        'NaN_percent': NaN_percent,

    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/CountFrequencyEncoding.html', context)


def CountFrequencyEncodingCalc(request, fName):
    df = get_df(fName)
    fName = currentFname(fName)

    clm_list = list(df)
    if request.method == 'POST':
        start = time.time()
        selected_cols = request.POST.getlist('CFCol')
        for selected_col in selected_cols:
            df_frequency_map = df[selected_col].value_counts().to_dict()
            df[selected_col] = df[selected_col].map(df_frequency_map)
            df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                   'processed/'+fName+'.csv'), index=False)
        df_new = get_df(fName)
        NaN_percent = get_NaN_percent(fName)
        clm_list_2 = list(df_new)
        NaN_percent = get_NaN_percent(fName)

        CF_list = list(set(clm_list_2) - set(selected_cols))

        end = time.time()
        print("Execution time of CountFrequencyEncoding", end - start)

        context = {
            'fName': fName,
            'cf_processing_list': CF_list,
            'cf_processed_list': selected_cols,
            'NaN_percent': NaN_percent,
            'status': 'Success',
            'message': 'Count Frequency Encoding was done on selected features.',
            'execution_time': end - start
        }

        context['undo_count'] = changesCount(fName)

        return render(request, 'FeatureEngineering/CountFrequencyEncoding.html', context)


def Normalization(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    normalization_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            normalization_list.append(clm)
        else:
            pass

    # labelled_list = []
    # for col_name in clm_list:
    #     if 'label' in col_name:
    #         labelled_list.append(col_name)
    #     else:
    #         pass
    context = {
        'fName': fName,
        'normalization_list': normalization_list,
        # 'labelled_list': labelled_list,
        'NaN_percent': NaN_percent,

    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/Normalization.html', context)


def NormalizationCalc(request, fName):
    df = get_df(fName)

    fName = currentFname(fName)

    start = time.time()

    if request.method == 'POST':
        normMethod = request.POST.get('normMethodSelected')
        selectedCols = request.POST.getlist('normCols')

        if normMethod == 'min-max':
            mini = int(request.POST.get('minNorm'))
            maxx = int(request.POST.get('maxNorm'))
            if mini != '' and maxx != '':
                for featureName in selectedCols:
                    df[featureName] = round(
                        (df[featureName] - mini) / (maxx - mini), 2)
            else:
                for featureName in selectedCols:
                    mini = min(df[featureName])
                    maxx = max(df[featureName])
                    df[featureName] = round(
                        (df[featureName] - mini) / (maxx - mini), 2)
            message = 'Normalization done using Min: ' + \
                str(mini)+' and Max: '+str(maxx)+' for range (0,1)'
            status = 'Success'
        elif normMethod == 'z-score':
            for featureName in selectedCols:
                mean = df[featureName].mean()
                df1 = abs(df[featureName] - mean)
                mad = sum(df1) / len(df1)
                df[featureName] = round((df[featureName] - mean) / mad, 2)
            message = 'Normalization done using Mean: ' + \
                str(mean)+' and Mean Absolute deviation: '+str(mad)
            status = 'Success'
        elif normMethod == 'decimal-scaling':
            for featureName in selectedCols:
                maxx = max(df[featureName])
                j = 1
                while maxx/j > 1:
                    j = j * 10
                df[featureName] = round(df[featureName] / j, 2)
            message = 'Normalization done using Decimal Scaling with value of ' + \
                str(j)
            status = 'Success'
        else:
            message = '*Please Select Atleast One Method for Normalization'
            status = 'Error'
    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                           'processed/'+fName+'.csv'), index=False)

    end = time.time()
    print("Execution time of Normalization", end - start)

    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    normalization_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            normalization_list.append(clm)
        else:
            pass

    context = {
        'fName': fName,
        'normalization_list': normalization_list,
        # 'labelled_list': labelled_list,
        'NaN_percent': NaN_percent,
        'message': message,
        'status': status,
        'execution_time': end - start
    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/Normalization.html', context)


def LogTransform(request, fName):

    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    log_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            log_list.append(clm)
        else:
            pass

    context = {
        'fName': fName,
        'log_list': log_list,
        'NaN_percent': NaN_percent,

    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/LogTransform.html', context)


def LogTransformCalc(request, fName):
    df = get_df(fName)

    fName = currentFname(fName)

    start = time.time()

    if request.method == "POST":
        selected_cols = request.POST.getlist('logCol')
    for col in selected_cols:
        df[col] = ((np.log(df[col])).replace(-np.inf, 0)).round(2)

    end = time.time()
    print("Execution time of LogTransform", end - start)

    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                           'processed/'+fName+'.csv'), index=False)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    log_list = []
    for clm in clm_list:
        dt = df[clm].dtype
        if dt == 'int64' or dt == 'float64':
            log_list.append(clm)
        else:
            pass

    context = {
        'fName': fName,
        'log_list': log_list,
        'NaN_percent': NaN_percent,
        'status': 'Success',
        'message': 'Log Transformation has been performed successfully',
        'execution_time': end - start
    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/LogTransform.html', context)

# getting dataframe
# =================


def get_df(fName):
    data_frame = pd.read_csv(os.path.join(settings.MEDIA_ROOT,
                                          'processed/'+fName+'.csv'), parse_dates=True)

    return data_frame


# Kurtosis
# ========
def kurtosis(fName):
    df = get_df(fName)
    df_kurtosis = df.kurt(axis=None, skipna=True).round(2)
    df_kurtosis_dict = df_kurtosis.to_dict()
    col = df_kurtosis_dict.keys()
    val = df_kurtosis_dict.values()
    kurtosis_list = zip(col, val)
    return kurtosis_list

# Skewness
# ========


def skewness(fName):
    df = get_df(fName)
    df_skewness = df.skew().round(2)
    df_skewness_dict = df_skewness.to_dict()
    val = df_skewness_dict.values()
    return val


# NaN Percentage
# ==============

def get_NaN(fName):
    df = get_df(fName)
    NaN_list = (df.isnull().sum() * 100 / len(df)).round(2)
    return NaN_list


# Mean
# =====

def get_mean(fName):
    df = get_df(fName)
    df_mean = df.mean().round(2)
    clm_list = list(df)
    percent = (df_mean * 100 / len(df)).round(2)
    mean_list = zip(clm_list, df_mean, percent)
    return mean_list


# Median
# ======

def get_median(fName):
    df = get_df(fName)
    df_median = df.median().round(2)
    clm_list = list(df)
    percent = (df_median * 100 / len(df)).round(2)
    median_list = zip(clm_list, df_median, percent)
    return median_list

# Standard Deviation
# ======


def get_std(fName):
    df = get_df(fName)
    df_std = df.std().round(2)
    clm_list = list(df)
    percent = (df_std * 100 / len(df)).round(2)
    std_list = zip(clm_list, df_std, percent)
    return std_list


# New Feature Generation
# ======================


def NewFeature(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    new_feature_list = []
    for clm in clm_list:
        if '_dt' in clm or 'date' in clm:
            new_feature_list.append(clm)
        else:
            pass

    context = {
        'fName': fName,
        'new_feature_list': new_feature_list,
        'NaN_percent': NaN_percent,

    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/NewFeature.html', context)


def NewFeatureCalc(request, fName):
    df = get_df(fName)

    fName = currentFname(fName)

    newly_generated_list = []

    start = time.time()

    if request.method == "POST":
        selected_cols = request.POST.getlist('newFeatureCol')

    for i in selected_cols:
        if '_dt' in i.lower() or 'date' in i.lower():
            df[i] = pd.to_datetime(df[i], dayfirst=True)
            newname = i.replace('date', '')
            newname = newname.replace('dt', '')
            df[newname+'_day'] = ((df[i].dropna()).dt.day).astype(int)
            newly_generated_list.append(newname+'_day')
            df[newname+'_month'] = ((df[i].dropna()).dt.month).astype(int)
            newly_generated_list.append(newname+'_month')
            df[newname+'_year'] = ((df[i].dropna()).dt.year).astype(int)
            newly_generated_list.append(newname+'_year')
            df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                   'processed/'+fName+'.csv'), index=False)
        else:
            pass

    end = time.time()
    print("Execution time of NewFeatureGeneration", end - start)

    clm_list = list(df)
    NaN_percent = get_NaN_percent(fName)
    new_feature_list = []
    for clm in clm_list:
        if '_dt' in clm or 'date' in clm:
            new_feature_list.append(clm)
        else:
            pass

    context = {
        'fName': fName,
        'new_feature_list': new_feature_list,
        'newly_generated_list': newly_generated_list,
        'NaN_percent': NaN_percent,
        'status': 'Success',
        'message': 'New features are generated successfully from the selected feature(s).',
        'execution_time': end - start
    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'FeatureEngineering/NewFeature.html', context)


# Download Original Dataset
# ==========================

def DownloadOriginal(request, fName):
    result_array = fName.rsplit('_', 1)
    fName = result_array[0]+'_0'
    file_path = os.path.join(settings.MEDIA_ROOT, 'processed/'+fName+'.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(
                fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + \
                os.path.basename(file_path)
            return response
    raise Http404


# Download Processed Dataset
# ==========================

def DownloadProcessed(request, fName):
    file_path = os.path.join(settings.MEDIA_ROOT, 'processed/'+fName+'.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(
                fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + \
                os.path.basename(file_path)
            return response
    raise Http404


# Download Dataset from External Source
# =====================================

def ExternalDataset(request, fName):
    return render(request, 'Dataset/DatasetCollection.html', context={'fName': fName})


# Remove Dataset
# ==============


def RemoveDataset(request, fName):
    original_file_path = os.path.join(
        settings.MEDIA_ROOT, 'original/'+fName+'.csv')
    processed_file_path = os.path.join(
        settings.MEDIA_ROOT, 'processed/'+fName+'.csv')
    if os.path.exists(original_file_path and processed_file_path):
        os.remove(original_file_path)
        os.remove(processed_file_path)
    context = {
        'status': 'Success',
        'message': 'Dataset Removed Successfully.'
    }

    return render(request, 'Dataset/Upload.html', context)


def fetchDataset(request, fName):
    context = getVisualization(fName)
    df = get_df(fName)
    chartLabel = fName
    # Overall columns
    categorical_clms_lst = []
    date_time_clms_lst = []
    numerical_clms_lst = []

    nan_clms = list(get_NaN(fName).to_dict().keys())
    nan_values = list(df.isnull().sum(axis=0))

    cols = list(df)

    for i in cols:
        if 'date' in i.lower() or 'dt' in i.lower() and 'day' not in i.lower() and 'month' not in i.lower() and 'year' not in i.lower():
            df[i] = pd.to_datetime(df[i], dayfirst=True)
            date_time_clms_lst.append(i)
            df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                   'processed/'+fName+'.csv'), index=False)
        elif df[i].dtypes == 'int64' or df[i].dtypes == 'float64':
            numerical_clms_lst.append(i)
        else:
            categorical_clms_lst.append(i)

    for date_time_col in date_time_clms_lst:
        df[date_time_col] = pd.to_datetime(df[date_time_col], dayfirst=True)

    cols_label = ['numberical-columns',
                  'categorical-columns', 'Datetime-columns']
    cols_data = [len(numerical_clms_lst), len(categorical_clms_lst), len(
        date_time_clms_lst)]

    # skewness
    df_skewness = df.skew().round(2)
    df_skewness_dict = df_skewness.to_dict()
    skew_col = list(df_skewness_dict.keys())
    skew_val = list(df_skewness_dict.values())

    # kurtosis
    df_kurtosis = df.kurt().round(2)
    df_kurtosis_dict = df_kurtosis.to_dict()
    kurt_col = list(df_kurtosis_dict.keys())
    kurt_val = list(df_kurtosis_dict.values())

    data = {
        "label": chartLabel,
        "skew_chartdata": skew_val,
        "kurt_chartdata": kurt_val,
        "skew_chartlabel": skew_col,
        "kurt_chartlabel": kurt_col,
        "cols_chartlabel": cols_label,
        "cols_chartdata": cols_data,
        "NaN_clms": nan_clms,
        "NaN_val": nan_values,

    }
    return JsonResponse(data)


def Visualize(request, fName):

    df = get_df(fName)

    start = time.time()

    clm_list = list(df)

    # skewness
    df_skewness = df.skew().round(2)
    df_skewness_dict = df_skewness.to_dict()
    skew_col = list(df_skewness_dict.keys())
    skew_val = list(df_skewness_dict.values())

    # heatmap restrict year colm
    for col in df.columns:
        if 'year' in col:
            del df[col]
        else:
            pass

    # correlation heatmap
    df123 = df.corr(numeric_only=False)
    heatmap_xy = list(df123)
    heatmap_z = df123.values.round(2)
    # print(heatmap_z)

    # categorical count frequency
    featureList = []
    countValues = []
    categorical_clm_list = []
    numerical_clm_list = []
    numerical_values = []
    for i in df.columns:
        if df[i].dtype == object:
            categorical_clm_list.append(i)
        else:
            numerical_clm_list.append(i)

    for i in categorical_clm_list:
        x = df[i].value_counts()
        x = x.sort_index(axis=0)
        featureList.append(x.index.tolist())
        countValues.append(list(x))

    for nu in numerical_clm_list:
        v = df[nu].head(10)
        numerical_values.append(v)

    if request.method == 'POST':

        df = get_df(fName)
        featureList = list(df)
        clm_list = []
        categorical_clm_list = []
        for i in list(df):
            if df[i].dtype == 'int64' or df[i].dtype == 'float64':
                clm_list.append(i)
            else:
                categorical_clm_list.append(i)
        numerical_clm_list = clm_list
        nan_percent = get_NaN_percent(fName)
        xFeature = ''
        yFeature = ''
        X_selected = ''
        Y_selected = ''
        colorFeature = ''
        featureValues = ''
        chart_type = request.POST.get('chartType')
        cont = ''
        xFeature = request.POST.get('param1')
        yFeature = request.POST.get('param2')
        colorFeature = request.POST.get('param2')
        if xFeature != '' and yFeature != '':
            x = list(df[xFeature])
            y = list(df[yFeature])

            df[xFeature] = df[xFeature].astype(
                str) + ' + ' + df[yFeature].astype(str)
            df = df[xFeature].value_counts()
            df = df.sort_index(axis=0)
            featureValues = df.index.tolist()
            count = list(df)
            X_selected = xFeature
            xFeature = xFeature + ' + ' + yFeature
            context = {
                'fName': fName,
                'clm_list': featureList,

                # skewness
                'skewness_col': skew_col,
                'skewness_val': skew_val,

                # categorical
                'featureList': featureList,
                'countValues': countValues,
                'categorical_clm_list': categorical_clm_list,
                # numerical
                'numerical_clm_list': numerical_clm_list,
                'numerical_values': numerical_values,
                # heatmap
                'heatmap_xy': heatmap_xy,
                'heatmap_z': heatmap_z,

                # Custom Chart
                'featureValues': featureValues,
                'count': count,
                'featureList': featureList,
                'featureName': xFeature,
                'xAxis': x,
                'yAxis': y,
                'Nan_percent': nan_percent,
                'customChartMsg': 'True',
                'custom_chart_status': 'True',
                'chart_type': chart_type,
                'x_selected': X_selected,
                'y_selected': yFeature,
                'numerical_clm_list': numerical_clm_list,
                'categorical_clm_list': categorical_clm_list,
            }
            context['undo_count'] = changesCount(fName)
            return render(request, 'CategoricalVisualize.html', context)
        if xFeature == '':
            xFeature = yFeature
        if xFeature == '':
            xFeature = colorFeature
        if xFeature != '':
            df = df[xFeature].value_counts()
        df = df.sort_index(axis=0)
        featureValues = df.index.tolist()
        count = list(df)
        mini = min(count)
        maxx = max(count)
        if yFeature != '':
            Y_selected = yFeature
            xFeature = xFeature + ' + ' + yFeature
        context = {
            'fName': fName,
            'clm_list': clm_list,

            # skewness
            'skewness_col': skew_col,
            'skewness_val': skew_val,

            # categorical
            'featureList': featureList,
            'countValues': countValues,
            'categorical_clm_list': categorical_clm_list,
            # numerical
            'numerical_clm_list': numerical_clm_list,
            'numerical_values': numerical_values,
            # heatmap
            'heatmap_xy': heatmap_xy,
            'heatmap_z': heatmap_z,

            # Custom Chart
            'featureValues': featureValues,
            'count': count,
            'featureList': featureList,
            'featureName': xFeature,
            'Nan_percent': nan_percent,
            'customChartMsg': 'True',
            'custom_chart_status': 'True',
            'chart_type': chart_type,
            'x_selected': xFeature,
            'y_selected': Y_selected,
            'numerical_clm_list': numerical_clm_list,
            'categorical_clm_list': categorical_clm_list,

        }
        context['undo_count'] = changesCount(fName)
        return render(request, 'CategoricalVisualize.html', context)

    end = time.time()
    print("Execution time of Visualization", end - start)

    context = {
        'fName': fName,
        'clm_list': clm_list,

        # skewness
        'skewness_col': skew_col,
        'skewness_val': skew_val,

        # categorical
        'featureList': featureList,
        'countValues': countValues,
        'categorical_clm_list': categorical_clm_list,
        # numerical
        'numerical_clm_list': numerical_clm_list,
        'numerical_values': numerical_values,
        # heatmap
        'heatmap_xy': heatmap_xy,
        'heatmap_z': heatmap_z,

        'execution_time': end - start
    }

    context['undo_count'] = changesCount(fName)

    return render(request, 'CategoricalVisualize.html', context)


def ChangeDtype(request, fName):
    df = get_df(fName)
    clm_list = list(df)
    dtype_list = df.dtypes
    changeDt_list = zip(clm_list, dtype_list)
    # Datatype Conversions
    if request.method == 'POST':
        customDataType = request.POST.get('datatype')
        selectedColumns = request.POST.getlist('selectedColumnsDt')

        if customDataType == 'datetime':
            for col in selectedColumns:
                df[col] = df[col].add_suffix('_date')
            df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                   'processed/'+fName+'.csv'), index=False)
            status = 'Success'
            message = 'Datatype Changed Succesfully.'
        elif customDataType == 'int':
            pass
        elif customDataType == 'float':
            pass
        elif customDataType == 'category':
            pass
        else:
            status = 'Error'
            message = '*Please Choose Datatype.'

        clm_list = list(df)
        dtype_list = df.dtypes
        changeDt_list = zip(clm_list, dtype_list)

        context = Overview(fName)
        context['status'] = status
        context['message'] = message

        return render(request, 'index.html', context)

    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                           'processed/'+fName+'.csv'), index=False)
    context = Overview(fName)
    return redirect('index.html')


def KNNImputation(request, fName):
    df = get_df(fName)
    cols = list(df)

    # nan values imputation using KNNImputer
    # =====================================

    imputer = KNNImputer(n_neighbors=2)
    df = pd.DataFrame(imputer.fit_transform(df), columns=cols)

    fName = currentFname(fName)
    df.to_csv(os.path.join(settings.MEDIA_ROOT,
                           'processed/'+fName+'.csv'), index=False)

    context = Overview(fName)
    context['undo_count'] = changesCount(fName)
    context['status'] = 'Success'
    context['message'] = 'NaN values filled by KNN method'
    return render(request, 'index.html', context)


def IterativeImputation(request, fName):
    df = get_df(fName)
    cols = list(df)
    features = []
    features = df.columns[0:-1]
    for feature in features:
        df[feature] = df[feature].replace(0, np.nan)
    imputer = IterativeImputer(BayesianRidge())
    df2 = pd.DataFrame((imputer.fit_transform(df)).round(2), columns=cols)
    fName = currentFname(fName)
    df2.to_csv(os.path.join(settings.MEDIA_ROOT, 'processed/' + fName + '.csv'),
               index=False)
    context = Overview(fName)
    context['undo_count'] = changesCount(fName)
    context['status'] = 'Success'
    context['message'] = 'NaN values filled by IterativeImputer method'
    return render(request, 'index.html', context)


def Iterative_Imputer_DecTreeReg(request, fName):
    df = get_df(fName)
    cols = list(df)
    features = []
    features = df.columns[0:-1]
    for feature in features:
        df[feature] = df[feature].replace(0, np.nan)
    imputer = IterativeImputer(DecisionTreeRegressor(max_features="sqrt",
                                                     random_state=0))
    df2 = pd.DataFrame((imputer.fit_transform(df)).round(2), columns=cols)
    fName = currentFname(fName)
    df2.to_csv(os.path.join(settings.MEDIA_ROOT, "processed/"+fName+".csv"),
               index=False)
    context = Overview(fName)
    context['undo_count'] = changesCount(fName)
    context["status"] = "Success"
    context["message"] = "NaN values filled by DecTreeReg method"
    return render(request, "index.html", context)


def Iterative_Imputer_ExtraTreesRegressor(request, fName):
    df = get_df(fName)
    cols = list(df)
    features = []
    features = df.columns[0:-1]
    for feature in features:
        df[feature] = df[feature].replace(0, np.nan)
    imputer = IterativeImputer(
        ExtraTreesRegressor(n_estimators=10, random_state=0))
    df2 = pd.DataFrame((imputer.fit_transform(df)).round(2), columns=cols)
    fName = currentFname(fName)
    df2.to_csv(os.path.join(settings.MEDIA_ROOT, 'processed/'+fName+'.csv'),
               index=False)
    context = Overview(fName)
    context['undo_count'] = changesCount(fName)
    context['status'] = 'Success'
    context['message'] = 'NaN values filled by DecTreeReg method'
    return render(request, 'index.html', context)


def handleRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)
    else:
        raise
