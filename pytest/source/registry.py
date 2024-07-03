import requests


class Registry:
    def get_users(self):
        response = requests.get("https://jsonplaceholder.typicode.com/users")
        # if response.ok:
        if response.status_code == 200:
            return response.json()

        raise requests.HTTPError("Something went wrong.")


if __name__ == "__main__":
    import json  # Just for formatting.

    users = Registry().get_users()
    print(json.dumps(users, indent=4))
