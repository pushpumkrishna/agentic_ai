from typing import Dict, Any
from flask import app as flask_app
from dotenv import load_dotenv
import os
import configparser
from backend.UC1.config.logging_lib import Logging

# Load environment variables from .env file
load_dotenv()


def read_config_file(filename: str = "config.ini") -> configparser.RawConfigParser:
    """
    Load configuration from an INI file and return the configparser object.

    :param filename: The name of the configuration file 'config.ini'.
    :return: ConfigParser object containing configuration data.
    """
    config_ = configparser.RawConfigParser()

    try:
        # Construct the file path using the current file's directory and the filename
        file_path = os.path.join(os.path.join(os.path.dirname(__file__)), filename)
        # Convert to a relative path
        relative_path = os.path.relpath(file_path, os.path.dirname(os.getcwd()))

        # Attempt to read the configuration file
        if os.path.exists(file_path):
            config_.read(file_path)
            Logging.info(f"Configuration loaded successfully from {relative_path}")
        else:
            raise FileNotFoundError(
                f"Configuration file {filename} not found at {relative_path}"
            )

    except FileNotFoundError as fnf_error:
        Logging.error(f"Error: {fnf_error}")
        raise  # Re-raise the error to notify the caller if needed
    except configparser.Error as config_error:
        Logging.error(f"Error parsing configuration file: {config_error}")
        raise  # Re-raise for the caller to handle
    except Exception as e:
        Logging.error(f"An unexpected error occurred: {e}")
        raise  # Re-raise unexpected exceptions

    return config_


config = read_config_file()
ENV = os.environ.get("FLASK_ENV", "local")


class AzureConfig:
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_BASE: str
    AZURE_OPENAI_API_TYPE: str
    AZURE_OPENAI_API_VERSION: str
    GPT_MODEL_35: str
    GPT_MODEL_PREVIEW: str
    GPT_MODEL_EMBEDDING: str
    GPT_MODEL_DEPLOYMENT_NAME: str
    GPT_MODEL_PREVIEW_DEPLOYMENT_NAME: str
    GPT_MODEL_EMBEDDING_DEPLOYMENT_NAME: str

    def __init__(self):
        self.AZURE_OPENAI_KEY = os.getenv(
            "AZURE_OPENAI_KEY", config.get(ENV, "AZURE_OPENAI_KEY")
        )
        self.AZURE_OPENAI_BASE = os.getenv(
            "AZURE_OPENAI_BASE", config.get(ENV, "AZURE_OPENAI_BASE")
        )
        self.AZURE_OPENAI_API_TYPE = os.getenv(
            "AZURE_OPENAI_API_TYPE", config.get(ENV, "AZURE_OPENAI_API_TYPE")
        )
        self.AZURE_OPENAI_API_VERSION = os.getenv(
            "AZURE_OPENAI_API_VERSION", config.get(ENV, "AZURE_OPENAI_API_VERSION")
        )
        self.GPT_MODEL_35 = os.getenv("GPT_MODEL_35", config.get(ENV, "GPT_MODEL_35"))
        self.GPT_MODEL_PREVIEW = os.getenv(
            "GPT_MODEL_PREVIEW", config.get(ENV, "GPT_MODEL_PREVIEW")
        )
        self.GPT_MODEL_EMBEDDING = os.getenv(
            "GPT_MODEL_EMBEDDING", config.get(ENV, "GPT_MODEL_EMBEDDING")
        )
        self.GPT_MODEL_DEPLOYMENT_NAME = os.getenv(
            "GPT_MODEL_DEPLOYMENT_NAME", config.get(ENV, "GPT_MODEL_DEPLOYMENT_NAME")
        )
        self.GPT_MODEL_PREVIEW_DEPLOYMENT_NAME = os.getenv(
            "GPT_MODEL_PREVIEW_DEPLOYMENT_NAME",
            config.get(ENV, "GPT_MODEL_PREVIEW_DEPLOYMENT_NAME"),
        )
        self.GPT_MODEL_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
            "GPT_MODEL_EMBEDDING_DEPLOYMENT_NAME",
            config.get(ENV, "GPT_MODEL_EMBEDDING_DEPLOYMENT_NAME"),
        )

    @property
    def get_azure_openai_key(self) -> str:
        """Get azure_openai_key
        :return: string
        """
        return self.AZURE_OPENAI_KEY

    @property
    def get_azure_openai_base(self) -> str:
        """Get azure_openai_base key
        :return: string
        """
        return self.AZURE_OPENAI_BASE

    @property
    def get_azure_openai_api_type(self) -> str:
        """Get azure_openai_api_type
        :return: string
        """
        return self.AZURE_OPENAI_API_TYPE

    @property
    def get_azure_openai_api_version(self) -> str:
        """Get azure_openai_api_version
        :return: string
        """
        return self.AZURE_OPENAI_API_VERSION

    @property
    def get_gpt_model_35(self) -> str:
        """Get gpt_model_35
        :return: string
        """
        return self.GPT_MODEL_35

    @property
    def get_gpt_model_preview(self) -> str:
        """Get gpt Vision model
        :return: string
        """
        return self.GPT_MODEL_PREVIEW

    @property
    def get_gpt_model_embedding(self) -> str:
        """Get gpt embedding model
        :return: string
        """
        return self.GPT_MODEL_EMBEDDING

    @property
    def get_gpt_model_deployment_name(self) -> str:
        """Get gpt model deployment name
        :return: string
        """
        return self.GPT_MODEL_DEPLOYMENT_NAME

    @property
    def get_gpt_model_preview_deployment_name(self) -> str:
        """Get gpt_model_preview_deployment_name
        :return: string
        """
        return self.GPT_MODEL_PREVIEW_DEPLOYMENT_NAME

    @property
    def get_gpt_model_embedding_deployment_name(self) -> str:
        """Get gpt_model_embedding_deployment_name
        :return: string
        """
        return self.GPT_MODEL_EMBEDDING_DEPLOYMENT_NAME


class Config(AzureConfig):
    ENV: str
    DEBUG: str
    LOGGING_TYPE: str

    def load_config(self):
        """
        Load configuration from environment variables and fallback to config file values.
        """
        self.DEBUG = os.getenv(
            "DEBUG", config.getboolean(ENV, "debug", fallback="false")
        )
        self.LOGGING_TYPE = os.getenv(
            "LOGGING_TYPE", config.get(ENV, "logging_type", fallback="ERROR")
        )

    def get_config(self) -> Dict[str, Any]:
        """Get config and turn it into a dict
        :return: dictionary object
        """
        return self.__dict__

    @staticmethod
    def read_config():
        """Load config from class itself
        :return: config class
        """
        c = Config()
        c.load_config()
        return c

    def __str__(self) -> str:
        """
        String representation of the Config object, converting all key-value pairs to strings.
        :return: String showing the configuration with all values as strings.
        """
        config_items = self.get_config()
        config_str = ", ".join(
            f"{key}={str(value)}" for key, value in config_items.items()
        )
        return f"Config({config_str})"


def config_from_ini(app: flask_app):
    """Append configuration from ini file to flask config
    # :param app: flask config object
    :return: None
    """
    cfg = Config()
    cfg.load_config()
    Logging.info("done")
    app.config.from_object(cfg)
