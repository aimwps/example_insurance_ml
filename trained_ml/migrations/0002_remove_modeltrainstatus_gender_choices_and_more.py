# Generated by Django 4.0.4 on 2022-05-08 11:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trained_ml', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='modeltrainstatus',
            name='gender_choices',
        ),
        migrations.RemoveField(
            model_name='modeltrainstatus',
            name='region_choices',
        ),
        migrations.AlterField(
            model_name='modeltrainstatus',
            name='rf_age',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='modeltrainstatus',
            name='rf_bmi',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='modeltrainstatus',
            name='rf_children',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='modeltrainstatus',
            name='rf_gender',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='modeltrainstatus',
            name='rf_is_smoker',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='modeltrainstatus',
            name='rf_region',
            field=models.BooleanField(default=False),
        ),
    ]
