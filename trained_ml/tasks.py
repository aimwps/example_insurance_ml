from celery import shared_task






@shared_task
def train_model_from_db(training_settings):
    print(training_settings)


    return "success"
