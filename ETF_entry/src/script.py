import json
import requests

class ETF_entry():
    """
    This class is used for send LINE notify message.
    """

    def __init__(self, user_list_path="userlist.json") -> None:
        with open(user_list_path, "r", encoding="utf-8") as f:
            self.users = json.loads(f.read())
            f.close()

    def send(self, name: str, msg: str) -> None:
        """
        Send LINE notify to single user.
        name: user name in userlist.json
        msg:  message you want to send.
        """
        api_key = self.users[name]

        url = 'https://notify-api.line.me/api/notify'
        headers = {
            'Authorization': 'Bearer ' + api_key
        }
        data = {
            'message': msg
        }
        requests.post(url, headers=headers, data=data)

    def send_all(self, msg: str) -> None:
        """
        Send LINE notify to all users.
        msg:  message you want to send.
        """
        for user_name in self.users.keys():
            self.send(user_name, msg)
