from django import forms


class TransactionForm(forms.Form):
    transaction_id = forms.IntegerField(
        label="Transaction Number",
        min_value=0,
        widget=forms.NumberInput(
            attrs={
                "placeholder": "Example: 100",
                "class": "form-control",
            }
        ),
    )


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Upload CSV File",
        widget=forms.ClearableFileInput(
            attrs={"accept": ".csv"}
        ),
    )
