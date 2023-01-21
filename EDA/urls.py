from django.urls import path, re_path
from . import views
from . import data_integration

urlpatterns = [

    path('', views.Upload, name='Upload'),


    # Dataset Integration
    re_path(r'^upload/(?P<fName>[-\w.]+\w{0,50})/$',
            data_integration.DatasetIntegrationUpload, name="DatasetIntegrationUpload"),
    path('integrate/',
         data_integration.Integrate, name="Integrate"),
    re_path(r'^IntegratedDataset/(?P<fName>[-\w.]+\w{0,50})/$',
            data_integration.Dataset, name='Dataset'),
    re_path(r'^IntegratedDatasetDownload/(?P<fName>[-\w.]+\w{0,50})/$',
            data_integration.DownloadIntegratedDataset, name='DownloadIntegratedDataset'),


    # Overview and Exploration
    re_path(r'^Home/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Home, name='Home'),
    re_path(r'^Explore/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Explore, name='Explore'),
    re_path(r'^Dataset/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Dataset, name='Dataset'),
    re_path(r'^OriginalDataset/(?P<fName>[-\w.]+\w{0,50})/$',
            views.OriginalDataset, name='OriginalDataset'),


    # Data Imputation
    re_path(r'^CompleteDropNan/(?P<fName>[-\w.]+\w{0,50})/$',
            views.CompleteDropNan, name='CompleteDropNan'),
    re_path(r'^AttrDropNan/(?P<fName>[-\w.]+\w{0,50})/$',
            views.AttrDropNan, name='AttrDropNan'),
    re_path(r'^AttrDropNanCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.AttrDropNanCalc, name='AttrDropNanCalc'),
    re_path(r'^AttrDropColCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.AttrDropColCalc, name='AttrDropColCalc'),
    re_path(r'^AttrFillNan/(?P<fName>[-\w.]+\w{0,50})/$',
            views.AttrFillNan, name='AttrFillNan'),
    re_path(r'^AttrFillNanCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.AttrFillNanCalc, name='AttrFillNanCalc'),
    re_path(r'^KNNImputation/(?P<fName>[-\w.]+\w{0,50})/$',
            views.KNNImputation, name="KNNImputation"),


    # removeUnwantedFeatures
    re_path(r'^removeUnwantedFeatures/(?P<fName>[-\w.]+\w{0,50})/$',
            views.RemoveUnwantedFeatures, name="RemoveUnwantedFeatures"),
    re_path(r'^removeUnwantedFeaturesCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.RemoveUnwantedFeaturesCalc, name="RemoveUnwantedFeaturesCalc"),

    # Iterative Imputation

    re_path(r'^IterativeImputation/(?P<fName>[-\w.]+\w{0,50})/$',
            views.IterativeImputation, name="IterativeImputation"),
    re_path(r'^Iterative_Imputer_DecTreeReg/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Iterative_Imputer_DecTreeReg, name="Iterative_Imputer_DecTreeReg"),
    re_path(r'^Iterative_Imputer_ExtraTreesRegressor/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Iterative_Imputer_ExtraTreesRegressor, name="Iterative_Imputer_ExtraTreesRegressor"),



    # Feature Engineering

    # Binning
    re_path(r'^Binning/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Binning, name="Binning"),
    re_path(r'^BinningCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.BinningCalc, name="BinningCalc"),
    # Label Encoding
    re_path(r'^LabelEncoding/(?P<fName>[-\w.]+\w{0,50})/$',
            views.LabelEncoding, name="LabelEncoding"),
    re_path(r'^LabelEncodingCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.LabelEncodingCalc, name="LabelEncodingCalc"),
    # One Hot Encoding
    re_path(r'^OneHotEncoding/(?P<fName>[-\w.]+\w{0,50})/$',
            views.OneHotEncoding, name="OneHotEncoding"),
    re_path(r'^OneHotEncodingCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.OneHotEncodingCalc, name="OneHotEncodingCalc"),
    # Ordinal Encoding
    re_path(r'^OrdinalEncoding/(?P<fName>[-\w.]+\w{0,50})/$',
            views.OrdinalEncoding, name="OrdinalEncoding"),
    re_path(r'^OrdinalEncodingCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.OrdinalEncodingCalc, name="OrdinalEncodingCalc"),
    # Binary Encoding
    re_path(r'^BinaryEncoding/(?P<fName>[-\w.]+\w{0,50})/$',
            views.BinaryEncoding, name="BinaryEncoding"),
    re_path(r'^BinaryEncodingCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.BinaryEncodingCalc, name="BinaryEncodingCalc"),
    # Count Frequency Encoding
    re_path(r'^CountFrequencyEncoding/(?P<fName>[-\w.]+\w{0,50})/$',
            views.CountFrequencyEncoding, name="CountFrequencyEncoding"),
    re_path(r'^CountFrequencyEncodingCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.CountFrequencyEncodingCalc, name="CountFrequencyEncodingCalc"),
    # Normalization
    re_path(r'^Normalization/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Normalization, name="Normalization"),
    re_path(r'^NormalizationCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.NormalizationCalc, name="NormalizationCalc"),
    # Log Transform
    re_path(r'^LogTransform/(?P<fName>[-\w.]+\w{0,50})/$',
            views.LogTransform, name="LogTransform"),
    re_path(r'^LogTransformCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.LogTransformCalc, name="LogTransformCalc"),

    # New Feature Generation
    re_path(r'^NewFeature/(?P<fName>[-\w.]+\w{0,50})/$',
            views.NewFeature, name="NewFeature"),
    re_path(r'^NewFeatureCalc/(?P<fName>[-\w.]+\w{0,50})/$',
            views.NewFeatureCalc, name="NewFeatureCalc"),


    # Download Processed
    re_path(r'^DownloadProcessed/(?P<fName>[-\w.]+\w{0,50})/$',
            views.DownloadProcessed, name="DownloadProcessed"),

    # Download Original
    re_path(r'^DownloadOriginal/(?P<fName>[-\w.]+\w{0,50})/$',
            views.DownloadOriginal, name="DownloadOriginal"),

    # Download External Dataset
    re_path(r'ExternalDataset/(?P<fName>[-\w.]+\w{0,50})/$',
            views.ExternalDataset, name="ExternalDataset"),

    # Undo
    re_path(r'^Undo/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Undo, name="Undo"),



    re_path(r'^Visualize/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Visualize, name='Visualize'),


    # Remove Processed
    re_path(r'^RemoveDataset/(?P<fName>[-\w.]+\w{0,50})/$',
            views.RemoveDataset, name="RemoveDataset"),
    re_path(r'^api/(?P<fName>[-\w.]+\w{0,50})/$',
            views.fetchDataset, name="fetchDataset"),
    re_path(r'^customChart/(?P<fName>[-\w.]+\w{0,50})/$',
            views.Visualize, name="customChart"),
    re_path(r'^ChangeDtypeColumn/(?P<fName>[-\w.]+\w{0,50})/$',
            views.ChangeDtype, name="ChangeDtype"),







]
