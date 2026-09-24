from django.urls import path
from . import views


urlpatterns = [

    path(
        "",
        views.home,
        name="home"
    ),

    path(
        "predict/",
        views.predict,
        name="predict"
    ),

    path(
        "upload/",
        views.upload_csv,
        name="upload"
    ),

    path(
        "search/",
        views.search_transaction,
        name="search_transaction"
    ),

    path(
        "dashboard/",
        views.dashboard,
        name="dashboard"
    ),

    path(
        "download-results/",
        views.download_fraud_results,
        name="download_fraud_results"
    ),

    path(
        "download-dataset/",
        views.download_dataset,
        name="download_dataset"
    ),
]