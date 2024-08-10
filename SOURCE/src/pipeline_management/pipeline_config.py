class PipelineConfig:
    def __init__(self):
        """
        Initialize the PipelineConfig with an empty configuration.
        """
        self.config = {}

    def set_config(self, key: str, value: any):
        """
        Set a configuration parameter.
        :param key: Configuration key
        :param value: Configuration value
        """
        self.config[key] = value

    def get_config(self, key: str) -> any:
        """
        Get a configuration parameter.
        :param key: Configuration key
        :return: Configuration value
        """
        return self.config.get(key, None)


if __name__ == "__main__":
    config = PipelineConfig()
    config.set_config("key", "value")
    print(config.get_config("key"))