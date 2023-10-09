from locust import HttpUser, task, between, events
from locust.env import Environment
import os
import random


images_folder = '/Users/mschulze/Downloads/load_test_data'


@events.init.add_listener
def on_locust_init(environment: Environment, **kwargs: int) -> None:
    environment.filenames = os.listdir(images_folder)


class QuickstartUser(HttpUser):
    wait_time = between(1,5)  #user waits rand 1-5 secs between requests

    @task
    def call_root_endpoint(self) -> None:
        self.client.get('/')
    
    # 3 is the weight for rand prob. So since 1 task hits root endpoint and 3 hit predict... 
    # 1/4 prob test root and 3/4 prob test predict
    @task(3) 
    def call_predict(self) -> None:
        filename = self.get_random_image_filename()
        image_path = f'{images_folder}/{filename}'
        self.client.post(
                '/predict',
                data={},
                files=[("file", (filename, open(image_path, 'rb'), "image/jpeg"))],
        )

    
    def get_random_image_filename(self) -> str:
        return random.choice(self.environment.filenames)


# run using this... locust -f locust_test.py
