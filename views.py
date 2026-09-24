import os
import joblib
import pandas as pd
import numpy as np

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render


# ============================================================
# PATHS
# ============================================================

BASE_DIR = settings.BASE_DIR

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "fraud_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "random_forest_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "random_forest_scaler.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "metrics.pkl"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "uploaded_results"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "fraud_analysis_results.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


# ============================================================
# LOAD MODEL FILES SAFELY
# ============================================================

def load_model_file(path):
    """
    Safely load a joblib/pickle file.
    Returns None if the file does not exist or cannot be loaded.
    """

    if not os.path.exists(path):
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


MODEL = load_model_file(MODEL_PATH)
SCALER = load_model_file(SCALER_PATH)
METRICS = load_model_file(METRICS_PATH)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    """
    Convert a value to float safely.
    """

    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    """
    Convert a value to integer safely.
    """

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def convert_metric_to_percentage(value):
    """
    Convert metrics such as:

        0.9995 -> 99.95
        99.95  -> 99.95
    """

    value = safe_float(value)

    if value <= 1:
        return round(value * 100, 2)

    return round(value, 2)


def get_metric(metrics, name, default=0):
    """
    Safely retrieve a metric from metrics.pkl.
    """

    if metrics is None:
        return default

    try:

        if isinstance(metrics, dict):

            if name in metrics:
                return metrics[name]

            for key in metrics.keys():

                if str(key).lower() == name.lower():
                    return metrics[key]

        return default

    except Exception:
        return default


# ============================================================
# FRAUD PROBABILITY
# ============================================================

def get_fraud_probability(model, X):
    """
    Return probability of class 1 = Fraud.
    """

    try:

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)

            classes = getattr(
                model,
                "classes_",
                []
            )

            for index, cls in enumerate(classes):

                try:

                    if int(cls) == 1:

                        return float(
                            probabilities[0][index]
                        )

                except Exception:
                    pass

            # Fallback
            if probabilities.shape[1] >= 2:

                return float(
                    probabilities[0][1]
                )

    except Exception:
        pass

    return 0.0


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):
    """
    Determine risk level based on fraud probability.
    """

    probability = safe_float(probability)

    percentage = probability * 100

    if percentage >= 70:
        return "HIGH"

    elif percentage >= 30:
        return "MEDIUM"

    return "LOW"


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(dataframe):
    """
    Prepare dataframe for Random Forest.

    Required features:

        Time
        V1 ... V28
        Amount
    """

    df = dataframe.copy()

    # --------------------------------------------------------
    # CHECK REQUIRED FEATURES
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # SELECT FEATURES IN EXACT MODEL ORDER
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # CONVERT TO NUMERIC
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        0
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # APPLY SCALER
    # --------------------------------------------------------

    if SCALER is not None:

        try:

            if hasattr(
                SCALER,
                "n_features_in_"
            ):

                scaler_features = (
                    SCALER.n_features_in_
                )

                # Scaler trained on all 30 features
                if scaler_features == len(
                    FEATURE_COLUMNS
                ):

                    return SCALER.transform(X)

                # Scaler trained only on Time and Amount
                elif scaler_features == 2:

                    X_scaled = X.copy()

                    scaled_values = (
                        SCALER.transform(
                            X_scaled[
                                ["Time", "Amount"]
                            ]
                        )
                    )

                    X_scaled[
                        ["Time", "Amount"]
                    ] = scaled_values

                    return X_scaled

        except Exception:
            pass

    # --------------------------------------------------------
    # RETURN RAW FEATURES
    # --------------------------------------------------------

    return X


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():
    """
    Extract Random Forest feature importance.

    Returns:
        labels
        values
        dictionary
    """

    labels = []
    values = []
    importance_dict = {}

    if MODEL is None:
        return labels, values, importance_dict

    try:

        if not hasattr(
            MODEL,
            "feature_importances_"
        ):
            return labels, values, importance_dict

        importances = MODEL.feature_importances_

        # ----------------------------------------------------
        # Match model importance values with feature names
        # ----------------------------------------------------

        if len(importances) != len(
            FEATURE_COLUMNS
        ):
            return labels, values, importance_dict

        importance_pairs = list(
            zip(
                FEATURE_COLUMNS,
                importances
            )
        )

        # ----------------------------------------------------
        # Sort highest to lowest
        # ----------------------------------------------------

        importance_pairs.sort(
            key=lambda item: item[1],
            reverse=True
        )

        # ----------------------------------------------------
        # Top 10
        # ----------------------------------------------------

        for feature, importance in (
            importance_pairs[:10]
        ):

            labels.append(feature)

            values.append(
                round(
                    float(importance) * 100,
                    4
                )
            )

            importance_dict[
                feature
            ] = round(
                float(importance) * 100,
                4
            )

        return (
            labels,
            values,
            importance_dict
        )

    except Exception:
        return (
            [],
            [],
            {}
        )


# ============================================================
# HOME
# ============================================================

def home(request):

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    total_transactions = 0
    fraudulent_transactions = 0
    normal_transactions = 0

    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:

        if os.path.exists(
            DATASET_PATH
        ):

            df = pd.read_csv(
                DATASET_PATH
            )

            total_transactions = len(df)

            if "Class" in df.columns:

                actual = pd.to_numeric(
                    df["Class"],
                    errors="coerce"
                ).fillna(0).astype(int)

                fraudulent_transactions = int(
                    (actual == 1).sum()
                )

                normal_transactions = (
                    total_transactions
                    - fraudulent_transactions
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # Fraud rate
    # --------------------------------------------------------

    fraud_rate = 0

    if total_transactions > 0:

        fraud_rate = round(
            (
                fraudulent_transactions
                / total_transactions
            ) * 100,
            2
        )

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    try:

        accuracy = convert_metric_to_percentage(
            get_metric(
                METRICS,
                "accuracy"
            )
        )

        precision = convert_metric_to_percentage(
            get_metric(
                METRICS,
                "precision"
            )
        )

        recall = convert_metric_to_percentage(
            get_metric(
                METRICS,
                "recall"
            )
        )

        f1_score = convert_metric_to_percentage(
            get_metric(
                METRICS,
                "f1_score"
            )
        )

    except Exception:
        pass

    context = {

        "total_transactions":
            total_transactions,

        "fraudulent_transactions":
            fraudulent_transactions,

        "normal_transactions":
            normal_transactions,

        "fraud_rate":
            fraud_rate,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1_score,
    }

    return render(
        request,
        "home.html",
        context
    )


# ============================================================
# DETECT FRAUD
# ============================================================

def predict(request):

    context = {

        "prediction": None,

        "is_fraud": False,

        "fraud_probability": 0,

        "risk_level": "LOW",

        "amount": 0,

        "transaction_id": "",

        "error": None,
    }

    # --------------------------------------------------------
    # Only process POST
    # --------------------------------------------------------

    if request.method == "POST":

        transaction_id = request.POST.get(
            "transaction_id",
            ""
        ).strip()

        amount = request.POST.get(
            "amount",
            "0"
        ).strip()

        context[
            "transaction_id"
        ] = transaction_id

        try:

            amount_value = safe_float(
                amount
            )

            context[
                "amount"
            ] = amount_value

            # ------------------------------------------------
            # Model check
            # ------------------------------------------------

            if MODEL is None:

                raise ValueError(
                    "Random Forest model could not be loaded."
                )

            # ------------------------------------------------
            # Build transaction
            #
            # For manual prediction:
            # V1-V28 are unknown, therefore 0.
            # ------------------------------------------------

            transaction = {

                "Time": 0,

                "Amount":
                    amount_value,
            }

            for feature in FEATURE_COLUMNS:

                if feature not in transaction:

                    transaction[
                        feature
                    ] = 0

            df = pd.DataFrame(
                [transaction]
            )

            # ------------------------------------------------
            # Prepare
            # ------------------------------------------------

            X = prepare_features(
                df
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction_value = MODEL.predict(
                X
            )[0]

            is_fraud = (
                int(prediction_value) == 1
            )

            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            probability = get_fraud_probability(
                MODEL,
                X
            )

            # ------------------------------------------------
            # Risk
            # ------------------------------------------------

            risk_level = get_risk_level(
                probability
            )

            # ------------------------------------------------
            # Context
            # ------------------------------------------------

            context[
                "prediction"
            ] = (
                "Fraudulent"
                if is_fraud
                else "Normal"
            )

            context[
                "is_fraud"
            ] = is_fraud

            context[
                "fraud_probability"
            ] = round(
                probability * 100,
                2
            )

            context[
                "risk_level"
            ] = risk_level

        except Exception as e:

            context[
                "error"
            ] = str(e)

    return render(
        request,
        "predict.html",
        context
    )


# ============================================================
# SEARCH TRANSACTION
# ============================================================

def search_transaction(request):

    context = {

        "transaction_id": "",

        "amount": None,

        "prediction": None,

        "is_fraud": False,

        "fraud_probability": 0,

        "risk_level": "LOW",

        "actual_class": None,

        "actual_label": None,

        "error": None,
    }

    # --------------------------------------------------------
    # Only search after form submission
    # --------------------------------------------------------

    if request.method != "POST":

        return render(
            request,
            "search.html",
            context
        )

    # --------------------------------------------------------
    # Transaction ID
    # --------------------------------------------------------

    transaction_id = request.POST.get(
        "transaction_id",
        ""
    ).strip()

    context[
        "transaction_id"
    ] = transaction_id

    if transaction_id == "":

        context[
            "error"
        ] = "Please enter a transaction ID."

        return render(
            request,
            "search.html",
            context
        )

    # --------------------------------------------------------
    # Dataset check
    # --------------------------------------------------------

    if not os.path.exists(
        DATASET_PATH
    ):

        context[
            "error"
        ] = (
            "Dataset not found at: "
            + DATASET_PATH
        )

        return render(
            request,
            "search.html",
            context
        )

    # --------------------------------------------------------
    # Model check
    # --------------------------------------------------------

    if MODEL is None:

        context[
            "error"
        ] = (
            "Random Forest model could not be loaded."
        )

        return render(
            request,
            "search.html",
            context
        )

    try:

        # ====================================================
        # READ DATASET
        # ====================================================

        df = pd.read_csv(
            DATASET_PATH
        )

        # ====================================================
        # REMOVE UNNAMED COLUMNS
        # ====================================================

        unnamed_columns = [

            column

            for column in df.columns

            if str(column).startswith(
                "Unnamed:"
            )
        ]

        if unnamed_columns:

            df = df.drop(
                columns=unnamed_columns
            )

        # ====================================================
        # POSSIBLE ID COLUMNS
        # ====================================================

        id_columns = [

            "TransactionID",

            "Transaction ID",

            "transaction_id",

            "transactionId",

            "ID",

            "id",
        ]

        found_id_column = None

        for column in id_columns:

            if column in df.columns:

                found_id_column = column

                break

        # ====================================================
        # FIND TRANSACTION
        # ====================================================

        transaction_row = None

        # ----------------------------------------------------
        # Dataset has actual ID column
        # ----------------------------------------------------

        if found_id_column:

            search_value = str(
                transaction_id
            ).strip()

            id_values = (
                df[
                    found_id_column
                ]
                .astype(str)
                .str.strip()
            )

            matches = df[
                id_values == search_value
            ]

            if matches.empty:

                context[
                    "error"
                ] = (
                    f"Transaction ID "
                    f"{transaction_id} "
                    f"was not found in the dataset."
                )

                return render(
                    request,
                    "search.html",
                    context
                )

            transaction_row = matches.iloc[0]

        # ----------------------------------------------------
        # Classic Kaggle dataset
        #
        # No TransactionID column.
        # Row number acts as transaction ID.
        # ----------------------------------------------------

        else:

            try:

                row_number = int(
                    transaction_id
                )

            except ValueError:

                context[
                    "error"
                ] = (
                    "This dataset does not contain "
                    "a TransactionID column. "
                    "Please enter a numeric transaction ID."
                )

                return render(
                    request,
                    "search.html",
                    context
                )

            if (
                row_number < 0
                or
                row_number >= len(df)
            ):

                context[
                    "error"
                ] = (
                    f"Transaction ID "
                    f"{row_number} "
                    f"is outside the dataset range. "
                    f"Valid IDs are 0 to {len(df) - 1}."
                )

                return render(
                    request,
                    "search.html",
                    context
                )

            transaction_row = df.iloc[
                row_number
            ]

        # ====================================================
        # ONE ROW DATAFRAME
        # ====================================================

        transaction_df = pd.DataFrame(
            [transaction_row]
        )

        # ====================================================
        # AMOUNT
        # ====================================================

        if "Amount" in transaction_df.columns:

            context[
                "amount"
            ] = safe_float(
                transaction_df.iloc[0][
                    "Amount"
                ]
            )

        # ====================================================
        # ACTUAL CLASS
        # ====================================================

        if "Class" in transaction_df.columns:

            actual_class = safe_int(
                transaction_df.iloc[0][
                    "Class"
                ]
            )

            context[
                "actual_class"
            ] = actual_class

            if actual_class == 1:

                context[
                    "actual_label"
                ] = "Fraudulent"

            else:

                context[
                    "actual_label"
                ] = "Normal"

        # ====================================================
        # REMOVE NON-MODEL COLUMNS
        # ====================================================

        columns_to_remove = [

            "Class",

            "TransactionID",

            "Transaction ID",

            "transaction_id",

            "transactionId",

            "ID",

            "id",
        ]

        model_df = transaction_df.drop(

            columns=[

                column

                for column in columns_to_remove

                if column in transaction_df.columns
            ],

            errors="ignore"
        )

        # ====================================================
        # CHECK REQUIRED FEATURES
        # ====================================================

        missing_features = [

            feature

            for feature in FEATURE_COLUMNS

            if feature not in model_df.columns
        ]

        if missing_features:

            context[
                "error"
            ] = (
                "The selected transaction is missing "
                "required model features: "
                + ", ".join(missing_features)
            )

            return render(
                request,
                "search.html",
                context
            )

        # ====================================================
        # PREPARE FEATURES
        # ====================================================

        X = prepare_features(
            model_df
        )

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction_value = MODEL.predict(
            X
        )[0]

        is_fraud = (
            int(prediction_value) == 1
        )

        # ====================================================
        # FRAUD PROBABILITY
        # ====================================================

        probability = get_fraud_probability(
            MODEL,
            X
        )

        # ====================================================
        # RISK LEVEL
        # ====================================================

        risk_level = get_risk_level(
            probability
        )

        # ====================================================
        # CONTEXT
        # ====================================================

        context[
            "prediction"
        ] = (
            "Fraudulent"
            if is_fraud
            else "Normal"
        )

        context[
            "is_fraud"
        ] = is_fraud

        context[
            "fraud_probability"
        ] = round(
            probability * 100,
            2
        )

        context[
            "risk_level"
        ] = risk_level

    except Exception as e:

        context[
            "error"
        ] = (
            "Unable to search transaction: "
            + str(e)
        )

    return render(
        request,
        "search.html",
        context
    )


# ============================================================
# UPLOAD CSV / EXCEL
# ============================================================

def upload_csv(request):

    context = {

        "success": False,

        "filename": None,

        "file_type": None,

        "total_transactions": 0,

        "fraud_count": 0,

        "normal_count": 0,

        "fraud_rate": 0,

        "fraud_amount": 0,

        "normal_amount": 0,

        "error": None,
    }

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    if request.method == "POST":

        uploaded_file = request.FILES.get(
            "file"
        )

        if not uploaded_file:

            context[
                "error"
            ] = (
                "Please select a CSV or Excel file."
            )

            return render(
                request,
                "upload.html",
                context
            )

        filename = uploaded_file.name

        extension = os.path.splitext(
            filename
        )[1].lower()

        # ----------------------------------------------------
        # CHECK FILE TYPE
        # ----------------------------------------------------

        if extension not in [
            ".csv",
            ".xlsx",
            ".xls"
        ]:

            context[
                "error"
            ] = (
                "Only CSV, XLSX and XLS files are supported."
            )

            return render(
                request,
                "upload.html",
                context
            )

        try:

            # =================================================
            # READ FILE
            # =================================================

            if extension == ".csv":

                df = pd.read_csv(
                    uploaded_file
                )

                file_type = "CSV"

            elif extension == ".xlsx":

                df = pd.read_excel(
                    uploaded_file,
                    engine="openpyxl"
                )

                file_type = "Excel XLSX"

            else:

                df = pd.read_excel(
                    uploaded_file
                )

                file_type = "Excel XLS"

            # =================================================
            # REMOVE UNNAMED COLUMNS
            # =================================================

            unnamed_columns = [

                column

                for column in df.columns

                if str(column).startswith(
                    "Unnamed:"
                )
            ]

            if unnamed_columns:

                df = df.drop(
                    columns=unnamed_columns
                )

            # =================================================
            # CHECK REQUIRED FEATURES
            # =================================================

            missing_columns = [

                column

                for column in FEATURE_COLUMNS

                if column not in df.columns
            ]

            if missing_columns:

                context[
                    "error"
                ] = (
                    "Uploaded file is missing required "
                    "columns: "
                    + ", ".join(missing_columns)
                )

                return render(
                    request,
                    "upload.html",
                    context
                )

            # =================================================
            # MODEL CHECK
            # =================================================

            if MODEL is None:

                context[
                    "error"
                ] = (
                    "Random Forest model could not be loaded."
                )

                return render(
                    request,
                    "upload.html",
                    context
                )

            # =================================================
            # PREPARE DATA
            # =================================================

            X = prepare_features(
                df
            )

            # =================================================
            # MODEL PREDICTION
            # =================================================

            predictions = MODEL.predict(
                X
            )

            df[
                "Prediction"
            ] = predictions

            # =================================================
            # FRAUD PROBABILITY
            # =================================================

            try:

                if hasattr(
                    MODEL,
                    "predict_proba"
                ):

                    probabilities = (
                        MODEL.predict_proba(X)
                    )

                    classes = getattr(
                        MODEL,
                        "classes_",
                        []
                    )

                    fraud_index = None

                    for index, cls in enumerate(
                        classes
                    ):

                        try:

                            if int(cls) == 1:

                                fraud_index = index

                                break

                        except Exception:
                            pass

                    if fraud_index is not None:

                        df[
                            "Fraud Probability"
                        ] = probabilities[
                            :,
                            fraud_index
                        ]

                    else:

                        df[
                            "Fraud Probability"
                        ] = 0.0

                else:

                    df[
                        "Fraud Probability"
                    ] = (
                        df[
                            "Prediction"
                        ].astype(float)
                    )

            except Exception:

                df[
                    "Fraud Probability"
                ] = 0.0

            # =================================================
            # TOTALS
            # =================================================

            total_transactions = len(df)

            fraud_count = int(
                (
                    df[
                        "Prediction"
                    ] == 1
                ).sum()
            )

            normal_count = (
                total_transactions
                - fraud_count
            )

            # =================================================
            # AMOUNTS
            # =================================================

            df[
                "Amount"
            ] = pd.to_numeric(
                df[
                    "Amount"
                ],
                errors="coerce"
            ).fillna(0)

            fraud_amount = float(
                df.loc[
                    df[
                        "Prediction"
                    ] == 1,
                    "Amount"
                ].sum()
            )

            normal_amount = float(
                df.loc[
                    df[
                        "Prediction"
                    ] == 0,
                    "Amount"
                ].sum()
            )

            # =================================================
            # FRAUD RATE
            # =================================================

            fraud_rate = 0

            if total_transactions > 0:

                fraud_rate = round(
                    (
                        fraud_count
                        / total_transactions
                    ) * 100,
                    2
                )

            # =================================================
            # SAVE RESULTS
            # =================================================

            os.makedirs(
                RESULTS_DIR,
                exist_ok=True
            )

            df.to_csv(
                RESULTS_PATH,
                index=False
            )

            # =================================================
            # CONTEXT
            # =================================================

            context[
                "success"
            ] = True

            context[
                "filename"
            ] = filename

            context[
                "file_type"
            ] = file_type

            context[
                "total_transactions"
            ] = total_transactions

            context[
                "fraud_count"
            ] = fraud_count

            context[
                "normal_count"
            ] = normal_count

            context[
                "fraud_rate"
            ] = fraud_rate

            context[
                "fraud_amount"
            ] = round(
                fraud_amount,
                2
            )

            context[
                "normal_amount"
            ] = round(
                normal_amount,
                2
            )

        except Exception as e:

            context[
                "error"
            ] = (
                "Unable to analyze file: "
                + str(e)
            )

    return render(
        request,
        "upload.html",
        context
    )


# ============================================================
# DOWNLOAD FRAUD RESULTS
# ============================================================

def download_fraud_results(request):

    if not os.path.exists(
        RESULTS_PATH
    ):

        raise Http404(
            "Fraud analysis results not found."
        )

    return FileResponse(
        open(
            RESULTS_PATH,
            "rb"
        ),
        as_attachment=True,
        filename="fraud_analysis_results.csv"
    )


# ============================================================
# DOWNLOAD DATASET
# ============================================================

def download_dataset(request):

    if not os.path.exists(
        DATASET_PATH
    ):

        raise Http404(
            "Dataset not found."
        )

    return FileResponse(
        open(
            DATASET_PATH,
            "rb"
        ),
        as_attachment=True,
        filename="fraud_data.csv"
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard(request):

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    total_transactions = 0

    fraudulent_transactions = 0

    normal_transactions = 0

    fraud_rate = 0

    total_amount = 0

    fraud_amount = 0

    normal_amount = 0

    average_transaction_amount = 0

    average_fraud_amount = 0

    average_normal_amount = 0

    # --------------------------------------------------------
    # Predicted values
    # --------------------------------------------------------

    predicted_fraud_transactions = 0

    predicted_normal_transactions = 0

    predicted_fraud_rate = 0

    predicted_fraud_amount = 0

    predicted_normal_amount = 0

    # ========================================================
    # HOURLY DATA
    # ========================================================

    hourly_labels = list(
        range(24)
    )

    hourly_normal_counts = [
        0
        for _ in range(24)
    ]

    hourly_fraud_counts = [
        0
        for _ in range(24)
    ]

    # ========================================================
    # MODEL METRICS
    # ========================================================

    accuracy = 0

    precision = 0

    recall = 0

    f1_score = 0

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    tn = 0

    fp = 0

    fn = 0

    tp = 0

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance_labels = []

    feature_importance_values = []

    feature_importance = {}

    (
        feature_importance_labels,
        feature_importance_values,
        feature_importance
    ) = get_feature_importance()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    try:

        if os.path.exists(
            DATASET_PATH
        ):

            df = pd.read_csv(
                DATASET_PATH
            )

            # ------------------------------------------------
            # Total transactions
            # ------------------------------------------------

            total_transactions = len(
                df
            )

            # ------------------------------------------------
            # Actual labels
            # ------------------------------------------------

            if "Class" in df.columns:

                df[
                    "Actual"
                ] = pd.to_numeric(
                    df[
                        "Class"
                    ],
                    errors="coerce"
                ).fillna(
                    0
                ).astype(int)

            elif "Prediction" in df.columns:

                df[
                    "Actual"
                ] = pd.to_numeric(
                    df[
                        "Prediction"
                    ],
                    errors="coerce"
                ).fillna(
                    0
                ).astype(int)

            else:

                df[
                    "Actual"
                ] = 0

            # ------------------------------------------------
            # Actual fraud / normal
            # ------------------------------------------------

            fraudulent_transactions = int(
                (
                    df[
                        "Actual"
                    ] == 1
                ).sum()
            )

            normal_transactions = (
                total_transactions
                - fraudulent_transactions
            )

            # ------------------------------------------------
            # Actual fraud rate
            # ------------------------------------------------

            if total_transactions > 0:

                fraud_rate = round(
                    (
                        fraudulent_transactions
                        / total_transactions
                    ) * 100,
                    2
                )

            # =================================================
            # AMOUNT ANALYSIS
            # =================================================

            if "Amount" in df.columns:

                df[
                    "Amount"
                ] = pd.to_numeric(
                    df[
                        "Amount"
                    ],
                    errors="coerce"
                ).fillna(0)

                total_amount = float(
                    df[
                        "Amount"
                    ].sum()
                )

                if total_transactions > 0:

                    average_transaction_amount = (
                        total_amount
                        / total_transactions
                    )

                fraud_amount = float(
                    df.loc[
                        df[
                            "Actual"
                        ] == 1,
                        "Amount"
                    ].sum()
                )

                normal_amount = float(
                    df.loc[
                        df[
                            "Actual"
                        ] == 0,
                        "Amount"
                    ].sum()
                )

                if fraudulent_transactions > 0:

                    average_fraud_amount = (
                        fraud_amount
                        / fraudulent_transactions
                    )

                if normal_transactions > 0:

                    average_normal_amount = (
                        normal_amount
                        / normal_transactions
                    )

            # =================================================
            # MODEL PREDICTIONS
            # =================================================

            if MODEL is not None:

                try:

                    model_df = df.drop(
                        columns=[
                            "Class",
                            "Prediction",
                            "Fraud Probability",
                            "Actual",
                            "TransactionID",
                            "Transaction ID",
                            "transaction_id",
                            "transactionId",
                            "ID",
                            "id"
                        ],
                        errors="ignore"
                    )

                    X = prepare_features(
                        model_df
                    )

                    predictions = MODEL.predict(
                        X
                    )

                    df[
                        "Model Prediction"
                    ] = predictions

                    # ------------------------------------------------
                    # Predicted fraud
                    # ------------------------------------------------

                    predicted_fraud_transactions = int(
                        (
                            df[
                                "Model Prediction"
                            ] == 1
                        ).sum()
                    )

                    predicted_normal_transactions = (
                        total_transactions
                        - predicted_fraud_transactions
                    )

                    if total_transactions > 0:

                        predicted_fraud_rate = round(
                            (
                                predicted_fraud_transactions
                                / total_transactions
                            ) * 100,
                            2
                        )

                    # ------------------------------------------------
                    # Predicted amounts
                    # ------------------------------------------------

                    if "Amount" in df.columns:

                        predicted_fraud_amount = float(
                            df.loc[
                                df[
                                    "Model Prediction"
                                ] == 1,
                                "Amount"
                            ].sum()
                        )

                        predicted_normal_amount = float(
                            df.loc[
                                df[
                                    "Model Prediction"
                                ] == 0,
                                "Amount"
                            ].sum()
                        )

                    # ------------------------------------------------
                    # 24-HOUR PREDICTED CHART
                    #
                    # Time is elapsed seconds.
                    # Convert to 24-hour buckets.
                    # ------------------------------------------------

                    if "Time" in df.columns:

                        df[
                            "Time"
                        ] = pd.to_numeric(
                            df[
                                "Time"
                            ],
                            errors="coerce"
                        ).fillna(0)

                        df[
                            "Hour"
                        ] = (
                            (
                                df[
                                    "Time"
                                ] // 3600
                            ) % 24
                        ).astype(int)

                        for hour in range(24):

                            hour_data = df[
                                df[
                                    "Hour"
                                ] == hour
                            ]

                            hourly_normal_counts[
                                hour
                            ] = int(
                                (
                                    hour_data[
                                        "Model Prediction"
                                    ] == 0
                                ).sum()
                            )

                            hourly_fraud_counts[
                                hour
                            ] = int(
                                (
                                    hour_data[
                                        "Model Prediction"
                                    ] == 1
                                ).sum()
                            )

                except Exception:

                    # If prediction fails, keep chart at zero.
                    pass

    except Exception:

        pass

    # ========================================================
    # MODEL METRICS
    # ========================================================

    try:

        if METRICS is not None:

            accuracy = convert_metric_to_percentage(
                get_metric(
                    METRICS,
                    "accuracy"
                )
            )

            precision = convert_metric_to_percentage(
                get_metric(
                    METRICS,
                    "precision"
                )
            )

            recall = convert_metric_to_percentage(
                get_metric(
                    METRICS,
                    "recall"
                )
            )

            f1_score = convert_metric_to_percentage(
                get_metric(
                    METRICS,
                    "f1_score"
                )
            )

    except Exception:

        pass

    # ========================================================
    # REAL CONFUSION MATRIX
    # ========================================================

    try:

        if (
            MODEL is not None
            and os.path.exists(DATASET_PATH)
        ):

            cm_df = pd.read_csv(
                DATASET_PATH
            )

            # ------------------------------------------------
            # Remove unnamed columns
            # ------------------------------------------------

            unnamed_columns = [

                column

                for column in cm_df.columns

                if str(column).startswith(
                    "Unnamed:"
                )
            ]

            if unnamed_columns:

                cm_df = cm_df.drop(
                    columns=unnamed_columns
                )

            # ------------------------------------------------
            # Actual labels
            # ------------------------------------------------

            if "Class" in cm_df.columns:

                y_actual = pd.to_numeric(
                    cm_df[
                        "Class"
                    ],
                    errors="coerce"
                ).fillna(
                    0
                ).astype(int)

                # ------------------------------------------------
                # Remove non-model columns
                # ------------------------------------------------

                cm_model_df = cm_df.drop(
                    columns=[
                        "Class",
                        "TransactionID",
                        "Transaction ID",
                        "transaction_id",
                        "transactionId",
                        "ID",
                        "id"
                    ],
                    errors="ignore"
                )

                # ------------------------------------------------
                # Prepare
                # ------------------------------------------------

                X_cm = prepare_features(
                    cm_model_df
                )

                # ------------------------------------------------
                # Predict
                # ------------------------------------------------

                y_predicted = MODEL.predict(
                    X_cm
                )

                # ------------------------------------------------
                # TN
                # ------------------------------------------------

                tn = int(
                    (
                        (y_actual == 0)
                        &
                        (y_predicted == 0)
                    ).sum()
                )

                # ------------------------------------------------
                # FP
                # ------------------------------------------------

                fp = int(
                    (
                        (y_actual == 0)
                        &
                        (y_predicted == 1)
                    ).sum()
                )

                # ------------------------------------------------
                # FN
                # ------------------------------------------------

                fn = int(
                    (
                        (y_actual == 1)
                        &
                        (y_predicted == 0)
                    ).sum()
                )

                # ------------------------------------------------
                # TP
                # ------------------------------------------------

                tp = int(
                    (
                        (y_actual == 1)
                        &
                        (y_predicted == 1)
                    ).sum()
                )

                print(
                    "CONFUSION MATRIX"
                )

                print(
                    f"TN: {tn}"
                )

                print(
                    f"FP: {fp}"
                )

                print(
                    f"FN: {fn}"
                )

                print(
                    f"TP: {tp}"
                )

    except Exception as e:

        print(
            "Confusion matrix calculation error:",
            e
        )

    # ========================================================
    # PEAK FRAUD HOUR
    # ========================================================

    peak_fraud_hour = 0

    peak_fraud_count = 0

    if hourly_fraud_counts:

        peak_fraud_count = max(
            hourly_fraud_counts
        )

        if peak_fraud_count > 0:

            peak_fraud_hour = (
                hourly_fraud_counts.index(
                    peak_fraud_count
                )
            )

    # ========================================================
    # PEAK NORMAL HOUR
    # ========================================================

    peak_normal_hour = 0

    peak_normal_count = 0

    if hourly_normal_counts:

        peak_normal_count = max(
            hourly_normal_counts
        )

        if peak_normal_count > 0:

            peak_normal_hour = (
                hourly_normal_counts.index(
                    peak_normal_count
                )
            )

    # ========================================================
    # FINAL CONTEXT
    # ========================================================

    context = {

        # ----------------------------------------------------
        # Dataset totals
        # ----------------------------------------------------

        "total_transactions":
            total_transactions,

        "fraudulent_transactions":
            fraudulent_transactions,

        "normal_transactions":
            normal_transactions,

        "fraud_rate":
            fraud_rate,

        # ----------------------------------------------------
        # Predicted totals
        # ----------------------------------------------------

        "predicted_fraud_transactions":
            predicted_fraud_transactions,

        "predicted_normal_transactions":
            predicted_normal_transactions,

        "predicted_fraud_rate":
            predicted_fraud_rate,

        # ----------------------------------------------------
        # Amounts
        # ----------------------------------------------------

        "total_amount":
            round(
                total_amount,
                2
            ),

        "fraud_amount":
            round(
                fraud_amount,
                2
            ),

        "normal_amount":
            round(
                normal_amount,
                2
            ),

        "predicted_fraud_amount":
            round(
                predicted_fraud_amount,
                2
            ),

        "predicted_normal_amount":
            round(
                predicted_normal_amount,
                2
            ),

        "average_transaction_amount":
            round(
                average_transaction_amount,
                2
            ),

        # ----------------------------------------------------
        # Alias for dashboard templates that use average_amount
        # ----------------------------------------------------

        "average_amount":
            round(
                average_transaction_amount,
                2
            ),

        "average_fraud_amount":
            round(
                average_fraud_amount,
                2
            ),

        "average_normal_amount":
            round(
                average_normal_amount,
                2
            ),

        # ----------------------------------------------------
        # Hourly charts
        # ----------------------------------------------------

        "hourly_labels":
            hourly_labels,

        "hourly_normal_counts":
            hourly_normal_counts,

        "hourly_fraud_counts":
            hourly_fraud_counts,

        # ----------------------------------------------------
        # Peak fraud
        # ----------------------------------------------------

        "peak_fraud_hour":
            peak_fraud_hour,

        "peak_fraud_count":
            peak_fraud_count,

        "peak_normal_hour":
            peak_normal_hour,

        "peak_normal_count":
            peak_normal_count,

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1_score,

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        # ----------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------

        "feature_importance":
            feature_importance,

        "feature_importance_labels":
            feature_importance_labels,

        "feature_importance_values":
            feature_importance_values,
    }

    return render(
        request,
        "dashboard.html",
        context
    )