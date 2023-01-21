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
import os
import csv
import pandas as pd
import numpy as np
import time


def getDf(fName):
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT,
                                  'integration/'+fName+'.csv'))
    return df


def getPrimaryKeys(secondary_ds, primary_ds):
    # find primary features
    secondary_df_clm_list_as_set = set(secondary_ds)
    intersection = secondary_df_clm_list_as_set.intersection(
        primary_ds)
    primary_keys = list(intersection)
    return primary_keys


def DatasetIntegrationUpload(request, fName):
    start = time.time()

    message = ''
    status = ''
    can_view_or_download = False
    secondary_df_clm_list = []
    primary_df_clm_list = []
    primary_features = []
    length_of_primary_keys = 0

    if request.method == "POST":
        secondary_ds = request.FILES['secondary_dataset']
        primary_ds = request.FILES['primary_dataset']

        arr1 = secondary_ds.name.split('.', 1)
        arr2 = primary_ds.name.split('.', 1)
        secondary_ds_name = arr1[0]
        primary_ds_name = arr2[0]
        extension1 = arr1[1]
        extension2 = arr2[1]
        fullName_ds1 = secondary_ds_name+'.'+extension1
        fullName_ds2 = primary_ds_name+'.'+extension2

        # Validating the uploaded file
        if extension1 == 'csv' and extension2 == 'csv':
            fs = FileSystemStorage()

            if os.path.exists(os.path.join(settings.MEDIA_ROOT, 'integration/')):
                shutil.rmtree(os.path.join(
                    settings.MEDIA_ROOT, 'integration/'), ignore_errors=False,
                    onerror=handleRemoveReadonly)

            file_path1 = os.path.join(
                settings.MEDIA_ROOT, 'integration/'+fullName_ds1)
            file_path2 = os.path.join(
                settings.MEDIA_ROOT, 'integration/'+fullName_ds2)
            file_path3 = os.path.join(
                settings.MEDIA_ROOT, 'integration/new_dataset.csv')

            if os.path.exists(file_path3):
                can_view_or_download = True
            else:
                can_view_or_download: False

            fs.save('integration/'+fullName_ds1, secondary_ds)
            fs.save('integration/'+fullName_ds2, primary_ds)

            secondary_ds = getDf(secondary_ds_name)
            primary_ds = getDf(primary_ds_name)
            secondary_df_clm_list = list(secondary_ds)
            primary_df_clm_list = list(primary_ds)

            # find primary features
            primary_features = getPrimaryKeys(
                secondary_df_clm_list, primary_df_clm_list)
            length_of_primary_keys = len(primary_features)

        else:
            message = 'Please upload only .CSV files'
            status = 'Unsupported File Format'

        end = time.time()
        context = {
            'secondary_ds_name': secondary_ds_name,
            'secondary_ds_clm_list': secondary_df_clm_list,
            'primary_ds_name': primary_ds_name,
            'primary_ds_clm_list': primary_df_clm_list,
            'primary_keys': primary_features,
            'length_of_primary_keys': length_of_primary_keys,
            'message': message,
            'status': status,
            'can_view_or_download': can_view_or_download,
            'fName': fName
        }
        return render(request, 'Dataset/DatasetCollection.html', context)


def Integrate(request):

    if request.method == 'POST':

        message = ""
        status = ""
        can_view_or_download = False

        joinType = request.POST.get('joinType')
        fName = request.POST.get('fName')
        secondary_ds_name = request.POST.get('secondary_ds_name')
        primary_ds_name = request.POST.get('primary_ds_name')
        secondary_df = getDf(secondary_ds_name)
        primary_df = getDf(primary_ds_name)
        secondary_df_clm_list = list(secondary_df)
        primary_df_clm_list = list(primary_df)
        primary_keys = getPrimaryKeys(
            secondary_df_clm_list, primary_df_clm_list)
        length_of_primary_keys = len(primary_keys)

        if joinType == 'prim':

            selected_primary_keys = request.POST.getlist('selectedPrimaryKey')
            inner_merged = pd.merge(secondary_df, primary_df, on=primary_keys)
            inner_merged.to_csv(os.path.join(settings.MEDIA_ROOT,
                                             'integration/new_dataset.csv'), index=False)
            status = "Success"
            message = "Integration has been done based on the selected PRIMARY KEY(S)"
            can_view_or_download = True

        elif joinType == 'fea':

            selected_features = request.POST.getlist('selectedFeatures')
            for i in selected_features:
                primary_df[i] = secondary_df[i]

            primary_df.to_csv(os.path.join(settings.MEDIA_ROOT,
                                           'integration/new_dataset.csv'), index=False)
            status = "Success"
            message = "Integration has been done based on the selected FEATURE(S)"
            can_view_or_download = True

        else:
            concatenated = pd.concat([secondary_df, primary_df], axis=1)
            concatenated.to_csv(os.path.join(settings.MEDIA_ROOT,
                                             'integration/new_dataset.csv'), index=False)
            status = "Success"
            message = "Integration has been done with the given DATASETS."
            can_view_or_download = True

        context = {
            'secondary_ds_name': secondary_ds_name,
            'secondary_ds_clm_list': secondary_df_clm_list,
            'primary_ds_name': primary_ds_name,
            'primary_ds_clm_list': primary_df_clm_list,
            'primary_keys': primary_keys,
            'length_of_primary_keys': length_of_primary_keys,
            'message': message,
            'status': status,
            'can_view_or_download': can_view_or_download,
            'fName': fName
        }

        return render(request, 'Dataset/DatasetCollection.html', context)


# View Dataset
# ============

def Dataset(request, fName):
    df = getDf(fName)
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
        'integrated_ds': True,
        'values': data,
    }

    return render(request, 'Dataset/Dataset.html', context)


# Download Integerated Dataset
# ============================

def DownloadIntegratedDataset(request, fName):

    file_path = os.path.join(settings.MEDIA_ROOT, 'integration/'+fName+'.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(
                fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + \
                os.path.basename(file_path)
            return response
    raise Http404


def handleRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)
    else:
        raise
