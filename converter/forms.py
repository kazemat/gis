# -*- coding: utf-8 -*-
from django import forms
from django.utils.translation import gettext as _


class UploadFileForm(forms.Form):
    delimiter = forms.CharField(max_length=1, help_text=_('enter a delimiter of csv file'), required=True)
    file = forms.FileField(allow_empty_file=False, required=True, help_text=_('Select the csv file'))


    class Meta:
        widgets = {
            'file': forms.FileInput(attrs={'accept': 'text/csv'})
        }
