import json


class Config:

    @classmethod
    def from_json_file(cls, file):
        config = Config()
        with open(file, "r") as f:
            data_dict = json.load(f)
        config.__dict__ = data_dict
        return config

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    config = Config.from_json_file("config.json")
    print(config)