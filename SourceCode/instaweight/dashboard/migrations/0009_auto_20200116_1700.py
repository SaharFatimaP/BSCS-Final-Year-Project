# Generated by Django 3.0 on 2020-01-16 12:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0008_auto_20200116_1658'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dailyweight',
            name='date_time',
            field=models.DateTimeField(verbose_name='Date Time'),
        ),
    ]
