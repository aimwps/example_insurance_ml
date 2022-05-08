from celery import shared_task






@shared_task
def train_model_from_db():
    print("whoop we called the shared task")
    return "success"
