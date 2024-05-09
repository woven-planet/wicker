import os
from typing import Any, Dict, Literal

import wandb

from wicker.core.config import get_config
from wicker.core.definitions import DatasetID
from wicker.core.storage import S3PathFactory


def version_dataset(
    dataset_name: str,
    dataset_version: str,
    entity: str,
    metadata: Dict[str, Any],
    dataset_backend: Literal["s3"] = "s3",
) -> None:
    """
    Version the dataset on Weights and Biases using the config parameters defined in wickerconfig.json.

    Args:
        dataset_name: The name of the dataset to be versioned
        dataset_version: The version of the dataset to be versioned
        entity: Who the run will belong to
        metadata: The metadata to be logged as an artifact, enforces dataclass for metadata documentation
        dataset_backend: The backend where the dataset is stored, currently only supports s3
    """
    # needs to first acquire and set wandb creds
    # WANDB_API_KEY, WANDB_BASE_URL
    # _set_wandb_credentials()

    # needs to init the wandb run, this is going to be a 'data' run
    dataset_run = wandb.init(project="dataset_curation", name=f"{dataset_name}_{dataset_version}", entity=entity)

    # grab the uri of the dataset to be versioned
    dataset_uri = _identify_s3_url_for_dataset_version(dataset_name, dataset_version, dataset_backend)

    # establish the artifact and save the dir/s3_url to the artifact
    data_artifact = wandb.Artifact(f"{dataset_name}_{dataset_version}", type="dataset")
    data_artifact.add_reference(dataset_uri, name="dataset")

    # save metadata dict to the artifact
    data_artifact.metadata["version"] = dataset_version
    data_artifact.metadata["s3_uri"] = dataset_uri
    for key, value in metadata.items():
        data_artifact.metadata[key] = value

    # save the artifact to the run
    dataset_run.log_artifact(data_artifact)  # type: ignore
    dataset_run.finish()  # type: ignore


def _set_wandb_credentials() -> None:
    """
    Acquire the weights and biases credentials and load them into the environment.

    This load the variables into the environment as ENV Variables for WandB to use,
    this function overrides the previously set wandb env variables with the ones specified in
    the wicker config if they exist.
    """
    # load the config
    config = get_config()

    # if the keys are present in the config add them to the env
    wandb_config = config.wandb_config
    for field in wandb_config.__dataclass_fields__:  # type: ignore
        attr = wandb_config.__getattribute__(field)
        if attr is not None:
            os.environ[str(field).upper()] = attr
        else:
            if str(field).upper() not in os.environ:
                raise EnvironmentError(
                    f"Cannot use W&B without setting {str(field.upper())}. "
                    f"Specify in either ENV or through wicker config file."
                )


def _identify_s3_url_for_dataset_version(
    dataset_name: str,
    dataset_version: str,
    dataset_backend: Literal["s3"] = "s3",
) -> str:
    """
    Identify and return the s3 url for the dataset and version specified in the backend.

    Args:
        dataset_name: name of the dataset to retrieve url
        dataset_version: version of the dataset to retrieve url
        dataset_backend: backend of the dataset to retrieve url

    Returns:
        The url pointing to the dataset on storage
    """
    schema_path = ""
    if dataset_backend == "s3":
        # needs to do the parsing work to fetch the correct s3 uri
        schema_path = S3PathFactory().get_dataset_assets_path(DatasetID(name=dataset_name, version=dataset_version))
    return schema_path
