from django.shortcuts import render, redirect, reverse, HttpResponse, HttpResponseRedirect
from django.http import HttpResponse
from django.contrib.auth import authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import auth, messages
from dashboard.models import *
import dashboard.graph_utils as grpu
from django.db.models import Q, QuerySet, Count
from .forms import cattle_form, EditProfileForm, FilterForm
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.core.paginator import Paginator
from django.views.generic.edit import UpdateView
from datetime import date, timedelta
from django.db.models.fields import DateField


app_name = 'dashboard'
# login url
login_url = "login"


def login(request):
    """
        Login View
    """
    error = ""
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        # checking if the user exist with this password
        user = auth.authenticate(username=username, password=password)
        # if exists
        if user:
            auth.login(request, user)
            user.save()
            return redirect(reverse("dashboard"))
        else:
            error = "Invalid username or password!"

    return render(request, f"{app_name}/pages/login.html", {'error': error})


@login_required(login_url=login_url)
def dashboard(request):

    breed_labels = Breed.objects.all()
    breed_counts = []
    for label in breed_labels:
        breed_counts.append(Cattle.objects.filter(breed__breed_name=label).count())

    healthy = Cattle.objects.filter(status_objects__status__status_value__exact='Standard').count()
    need_care = Cattle.objects.filter(Q(status_objects__status__status_value__exact='Overweight') | Q(
        status_objects__status__status_value__exact='Underweight')).count()
    care = []
    care.append(need_care)
    care.append(healthy)
    need_care_lst = Cattle.objects.filter(Q(status_objects__status__status_value__exact='Overweight') | Q(
        status_objects__status__status_value__exact='Underweight'))
    if(len(need_care_lst) > 3):
        top_cattle = []
        for i in range(0, 3):
            top_cattle = top_cattle + ([need_care_lst[i]])
        need_care_lst = top_cattle

    cattle_id_lst = []
    rf_id_lst = []
    ear_tag_lst = []
    breed_lst = []
    img_lst = []

    for i in need_care_lst:
        cattle_id_lst = cattle_id_lst + [i.id]
        rf_id_lst = rf_id_lst + [i.rf_id]
        ear_tag_lst = ear_tag_lst + [i.ear_tag]
        breed_lst = breed_lst + [i.breed]
        img_lst = img_lst + [i.image]

    lst = []
    for i in range(0, 3):
        lst = lst + ([[cattle_id_lst[i], rf_id_lst[i], ear_tag_lst[i], breed_lst[i], img_lst[i]]])

    context = {
        "total_cattle": Cattle.objects.all().count(),
        "male_distribution": Cattle.objects.filter(gender__iexact="Male").count(),
        "female_distribution": Cattle.objects.filter(gender__iexact="Female").count(),
        "underweight_status": Cattle.objects.filter(status_objects__status__status_value__exact='Underweight').count(),
        "normal_status": Cattle.objects.filter(status_objects__status__status_value__exact='Standard').count(),
        "overweight_status": Cattle.objects.filter(status_objects__status__status_value__exact='Overweight').count(),
        "sold_status": Cattle.objects.filter(status_objects__status__status_value__exact='Sold').count(),
        "dead_status": Cattle.objects.filter(status_objects__status__status_value__exact='Dead').count(),
        "breed_labels": breed_labels,
        "breed_counts": breed_counts,
        "care": care,
        "need_care": need_care,
        "rf_id_lst": rf_id_lst,
        "ear_tag_lst": ear_tag_lst,
        "breed_lst": breed_lst,
        "need_care_lst": need_care_lst,
        "lst": lst
    }

    return render(request, f'{app_name}/pages/dashboard.html', context=context)


@login_required(login_url=login_url)
def cattle_management(request):
    form = cattle_form()
    if request.method == 'POST' and request.is_ajax():
        print(request.FILES)
        form = cattle_form(request.POST, files=request.FILES)

        if form.is_valid():
            form.save()
            form = cattle_form()

            return render(request, f'{app_name}/widgets/add_cattle_form.html', context={'form': form, "message": "Cattle Added Successfuly"})
        else:
            return render(request, f'{app_name}/widgets/add_cattle_form.html', context={'form': form}, status=400)

    cattles = None

    filter = FilterForm(request.GET)
    cattles = filter.FilterQuery()

    page_number = request.GET.get('page', 1)

    # paginations
    paginator = Paginator(cattles, 25)
    cattles = paginator.get_page(page_number)
    context = {
        "cattles": cattles,
        'form': form,
        'filter_form': filter
    }

    return render(request, f'{app_name}/pages/cattle-management.html', context=context)


@login_required(login_url=login_url)
def cattle_view(request, id):
    daily_weights = DailyWeight.objects.filter(cattle__id=id)
    dates = []
    weights = []
    for cattles in daily_weights:
        weights.append(cattles.weight)
        date = str(cattles.date_time.date())
        dates.append(date)

    cattle_detail = Cattle.objects.get(id=id)
    form = cattle_form(instance=cattle_detail)

    if request.method == 'POST':
        form = cattle_form(request.POST, files=request.FILES, instance=cattle_detail)

        if form.is_valid():
            form.save()
            return cattle_management(request)
        else:
            context = {
                "form": form,
                "daily_weights": daily_weights,
                "cattle": Cattle.objects.get(id=id),
                "dates": dates,
                "weights": weights
            }
            return render(request, f'{app_name}/widgets/cattle_details_modal_body.html', context=context)
    else:
        page_number = request.GET.get('page', 1)
        # paginations
        paginator = Paginator(daily_weights, 10)
        daily_weights = paginator.get_page(page_number)
        context = {
            "form": form,
            "daily_weights": daily_weights,
            "cattle": Cattle.objects.get(id=id),
            "dates": dates,
            "weights": weights
        }
        return render(request, f'{app_name}/widgets/cattle_details_modal_body.html', context=context)


@login_required(login_url=login_url)
def daily_view(request, id):
    detail_daily_weight = DailyWeight.objects.get(id=id)

    context = {
        'detail_daily_weight': detail_daily_weight,
    }

    return render(request, f'{app_name}/widgets/daily_weight_detail_page.html', context=context)


@login_required(login_url=login_url)
def log_weight(request, id):
    print(request.POST)
    if request.method == "POST":
        cattle = Cattle.objects.get(id=id)
        weight = request.POST.get("weight", 0)
        heart_girth = request.POST.get("heart_girth", 0)
        diagonal_len = request.POST.get("diagonal_len", 0)
        daily_weight = DailyWeight.objects.update_or_create(cattle=cattle, date_time__date=date.today(), defaults={
                                                            "cattle": cattle, "weight": weight, "heart_girth": heart_girth, 'diagonal_len': diagonal_len})
        return redirect(reverse("cattle_view", args=[id]))


@login_required(login_url=login_url)
def settings(request):
    if request.method == 'POST':
        profile_form_obj = EditProfileForm(request.POST)
        password_form_obj = PasswordChangeForm(request.POST)

        if "profile_form" in request.POST:
            profile_form_obj = EditProfileForm(request.POST, instance=request.user)
            if profile_form_obj.is_valid():
                user = profile_form_obj.save()

            context = {'profile_form': profile_form_obj, 'password_form': password_form_obj}
            return render(request, f'{app_name}/pages/settings.html', context=context)
        elif 'password_form' in request.POST:
            password_form_obj = PasswordChangeForm(request.user, request.POST)
            if password_form_obj.is_valid():
                user = password_form_obj.save()
                return redirect(reverse('login'))

            context = {'profile_form': profile_form_obj, 'password_form': password_form_obj}
            return render(request, f'{app_name}/pages/settings.html', context=context)
    else:
        profile_form_obj = EditProfileForm(instance=request.user)
        password_form_obj = PasswordChangeForm(request.user)
        return render(request, f'{app_name}/pages/settings.html', {'profile_form': profile_form_obj, 'password_form': password_form_obj})
