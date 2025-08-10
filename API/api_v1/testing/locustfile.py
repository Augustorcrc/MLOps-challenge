from locust import HttpUser, task, between
import uuid, random

REFS = [
    "John 3:16", "Genesis 1:1", "Psalm 23:1", "Romans 8:28",
    "Proverbs 3:5", "Matthew 6:33", "John 1:1"
]

class APIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def post_verse(self):
        headers = {"X-Request-ID": str(uuid.uuid4())}
        ref = random.choice(REFS)
        self.client.post("/verse", json={"reference": ref}, headers=headers)

    @task(1)
    def get_top3(self):
        self.client.get("/top")

# nota: Ejecutar locust -f locustfile.py --host http://localhost:8080