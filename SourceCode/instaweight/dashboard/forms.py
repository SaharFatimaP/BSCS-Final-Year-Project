from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from dashboard.models import *


class cattle_form(forms.ModelForm):

    class Meta:
        model = Cattle
        fields = ('rf_id', 'ear_tag', 'gender', 'breed', 'birth_date', 'image')
        widgets = {
            'birth_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'ear_tag': forms.TextInput(attrs={'class': 'form-control'}),
            'rf_id': forms.TextInput(attrs={'class': 'form-control'}),
            'breed': forms.Select(attrs={'class': 'form-control'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'gender': forms.Select(attrs={'class': 'form-control'})
        }


class EditProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('username', 'email')
        widgets = {

            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'})
        }


gender_list = [
    ('', 'All GENDER'),
    ('Male', 'Male'),
    ('Female', 'Female'),
]


class FilterForm(forms.Form):
    search = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Search Cattle'}), max_length=250, required=False)
    gender = forms.ChoiceField(choices=gender_list, widget=forms.Select(
        attrs={'class': 'form-control'}), required=False)
    status = forms.ModelChoiceField(Status.objects.all(), widget=forms.Select(
        attrs={'class': 'form-control'}), required=False, empty_label='All STATUS')

    def FilterQuery(self):
        self.is_valid()
        search_text = self.cleaned_data['search']
        gender = self.cleaned_data['gender']
        status = self.cleaned_data['status']

        # print(search_text, gender, status)

        querySet = Cattle.objects.all().order_by('rf_id')

        if not search_text == '':
            querySet = querySet.filter(Q(rf_id__iexact=search_text) | Q(
                ear_tag__iexact=search_text) | Q(gender__iexact=search_text))

        if gender:
            querySet = querySet.filter(gender__iexact=gender)

        if status:
            querySet = querySet.filter(status_objects__status=status)
        return querySet
